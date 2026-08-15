#!/usr/bin/env bash

set -Eeuo pipefail

diagnosis_root="hunyuan_groupnorm_diagnosis"
diagnosis_bad_commit="422d8fd8f"
diagnosis_fixed_commit="b01721535"
diagnosis_baseline="${1:-tests/assets/hunyuan/hunyuan_baseline.png}"
diagnosis_run_id="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
diagnosis_devices="${HUNYUAN_GROUPNORM_DEVICES:-0,1,2,3}"

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: run_diagnosis.sh [baseline-image] [run-id]

Environment:
  HUNYUAN_GROUPNORM_DEVICES  Four physical GPU IDs. Default: 0,1,2,3

All generated files are saved under:
  hunyuan_groupnorm_diagnosis/artifacts/runs/<run-id>
EOF
    exit 0
fi

if [[ ! -d .git || ! -f pyproject.toml ]]; then
    echo "Run this script from the vLLM Omni repository root." >&2
    exit 1
fi

if [[ ! "${diagnosis_run_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Run ID may contain only letters, numbers, dots, underscores, and hyphens." >&2
    exit 1
fi

if [[ ! -f "${diagnosis_baseline}" ]]; then
    echo "Baseline image does not exist: ${diagnosis_baseline}" >&2
    exit 1
fi

diagnosis_artifact_dir="${diagnosis_root}/artifacts/runs/${diagnosis_run_id}"
diagnosis_bad_source="${diagnosis_artifact_dir}/sources/bad"
diagnosis_fixed_source="${diagnosis_artifact_dir}/sources/fixed"
diagnosis_probe_source="${diagnosis_artifact_dir}/sources/probe"
diagnosis_tensor_dir="${diagnosis_artifact_dir}/tensor_dumps"

if [[ -e "${diagnosis_artifact_dir}" ]]; then
    echo "Run artifact directory already exists: ${diagnosis_artifact_dir}" >&2
    exit 1
fi

mkdir -p \
    "${diagnosis_artifact_dir}/bad" \
    "${diagnosis_artifact_dir}/fixed" \
    "${diagnosis_artifact_dir}/probe" \
    "${diagnosis_tensor_dir}"

exec > >(tee "${diagnosis_artifact_dir}/runner.log") 2>&1

diagnosis_export_source() {
    local diagnosis_source_path="$1"
    local diagnosis_source_commit="$2"
    local diagnosis_expected_commit

    diagnosis_expected_commit="$(git rev-parse "${diagnosis_source_commit}^{commit}")"

    if [[ -e "${diagnosis_source_path}" ]]; then
        echo "Source path already exists: ${diagnosis_source_path}" >&2
        exit 1
    fi

    mkdir -p "${diagnosis_source_path}"
    git archive "${diagnosis_expected_commit}" \
        | tar -x -C "${diagnosis_source_path}"
    printf "%s\n" "${diagnosis_expected_commit}" \
        > "${diagnosis_source_path}/.hunyuan_groupnorm_commit"
}

diagnosis_run_hunyuan() {
    local diagnosis_source_path="$1"
    local diagnosis_output_path="$2"
    local diagnosis_log_path="$3"
    local diagnosis_steps="$4"

    HIP_VISIBLE_DEVICES="${diagnosis_devices}" \
    VLLM_ROCM_USE_AITER=1 \
    VLLM_LOGGING_LEVEL=INFO \
    PYTHONPATH="${diagnosis_source_path}" \
    python \
        "${diagnosis_source_path}/examples/offline_inference/hunyuan_image3/end2end.py" \
        --model tencent/HunyuanImage-3.0-Instruct \
        --modality text2img \
        --deploy-config "${diagnosis_source_path}/vllm_omni/deploy/hunyuan_image3_dit.yaml" \
        --prompts "A brown and white dog is running on the grass." \
        --output "${diagnosis_output_path}" \
        --steps "${diagnosis_steps}" \
        --guidance-scale 2.5 \
        --seed 42 \
        --height 1024 \
        --width 1024 \
        --bot-task none \
        --sys-type en_unified \
        --init-timeout 900 \
        --enforce-eager \
        2>&1 | tee "${diagnosis_log_path}"
}

echo "Recording the environment."
{
    HIP_VISIBLE_DEVICES="${diagnosis_devices}" python - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution

import aiter
import torch

print("torch", torch.__version__)
print("hip", torch.version.hip)
print("aiter", aiter.__file__)
try:
    aiter_distribution = distribution("amd-aiter")
except PackageNotFoundError:
    print("aiter_version", "unknown")
    print("aiter_distribution_location", "unknown")
else:
    print("aiter_version", aiter_distribution.version)
    print("aiter_distribution_location", aiter_distribution.locate_file(""))
print("device_count", torch.cuda.device_count())

if torch.version.hip is None:
    raise RuntimeError("The active Python environment is not a ROCm build.")
if torch.cuda.device_count() < 4:
    raise RuntimeError("The diagnosis requires four visible GPUs.")
PY
    printf "vllm_omni "
    git rev-parse HEAD
    printf "devices %s\n" "${diagnosis_devices}"
    printf "baseline %s\n" "${diagnosis_baseline}"
} | tee "${diagnosis_artifact_dir}/environment.txt"

echo "Saving the runtime revision difference."
git diff --name-only "${diagnosis_bad_commit}..${diagnosis_fixed_commit}" \
    | tee "${diagnosis_artifact_dir}/revision_files.txt"
git diff "${diagnosis_bad_commit}..${diagnosis_fixed_commit}" -- \
    vllm_omni/platforms/rocm/patch/worker/patch_groupnorm.py \
    | tee "${diagnosis_artifact_dir}/groupnorm_runtime.diff"

echo "Exporting the bad and fixed source snapshots."
{
    diagnosis_export_source "${diagnosis_bad_source}" "${diagnosis_bad_commit}"
    diagnosis_export_source "${diagnosis_fixed_source}" "${diagnosis_fixed_commit}"
} | tee "${diagnosis_artifact_dir}/source_setup.log"

echo "Running the bad revision."
diagnosis_run_hunyuan \
    "${diagnosis_bad_source}" \
    "${diagnosis_artifact_dir}/bad/images" \
    "${diagnosis_artifact_dir}/bad/run.log" \
    50

echo "Running the fixed revision."
diagnosis_run_hunyuan \
    "${diagnosis_fixed_source}" \
    "${diagnosis_artifact_dir}/fixed/images" \
    "${diagnosis_artifact_dir}/fixed/run.log" \
    50

echo "Comparing both images with the baseline."
MPLCONFIGDIR="${diagnosis_root}/artifacts/matplotlib_cache" \
python "${diagnosis_root}/compare_images.py" \
    --baseline "${diagnosis_baseline}" \
    --bad "${diagnosis_artifact_dir}/bad/images/output_0_0.png" \
    --fixed "${diagnosis_artifact_dir}/fixed/images/output_0_0.png" \
    2>&1 | tee "${diagnosis_artifact_dir}/image_comparison.txt"

echo "Exporting a separate source snapshot for the live GroupNorm probe."
{
    diagnosis_export_source "${diagnosis_probe_source}" "${diagnosis_bad_commit}"
    git apply --check --directory="${diagnosis_probe_source}" \
        < "${diagnosis_root}/diagnostic_groupnorm.patch"
    git apply --directory="${diagnosis_probe_source}" \
        < "${diagnosis_root}/diagnostic_groupnorm.patch"
    diagnosis_patch_diff_status=0
    git diff --no-index -- \
        "${diagnosis_bad_source}/vllm_omni/platforms/rocm/patch/worker/patch_groupnorm.py" \
        "${diagnosis_probe_source}/vllm_omni/platforms/rocm/patch/worker/patch_groupnorm.py" \
        || diagnosis_patch_diff_status="$?"
    if (( diagnosis_patch_diff_status > 1 )); then
        exit "${diagnosis_patch_diff_status}"
    fi
} 2>&1 | tee "${diagnosis_artifact_dir}/probe/applied_patch.log"

echo "Running the live GroupNorm probe."
set +e
HIP_VISIBLE_DEVICES="${diagnosis_devices}" \
VLLM_ROCM_USE_AITER=1 \
VLLM_LOGGING_LEVEL=INFO \
TORCH_SHOW_CPP_STACKTRACES=1 \
HUNYUAN_GROUPNORM_DIAG_DIR="${diagnosis_tensor_dir}" \
PYTHONPATH="${diagnosis_probe_source}" \
python \
    "${diagnosis_probe_source}/examples/offline_inference/hunyuan_image3/end2end.py" \
    --model tencent/HunyuanImage-3.0-Instruct \
    --modality text2img \
    --deploy-config "${diagnosis_probe_source}/vllm_omni/deploy/hunyuan_image3_dit.yaml" \
    --prompts "A brown and white dog is running on the grass." \
    --output "${diagnosis_artifact_dir}/probe/images" \
    --steps 1 \
    --guidance-scale 2.5 \
    --seed 42 \
    --height 1024 \
    --width 1024 \
    --bot-task none \
    --sys-type en_unified \
    --init-timeout 900 \
    --enforce-eager \
    2>&1 | tee "${diagnosis_artifact_dir}/probe/run.log"
diagnosis_probe_status="${PIPESTATUS[0]}"
set -e
printf "probe_exit_status=%s\n" "${diagnosis_probe_status}" \
    | tee "${diagnosis_artifact_dir}/probe/status.txt"

shopt -s nullglob
diagnosis_dump_files=("${diagnosis_tensor_dir}"/groupnorm-mismatch-*.pt)
shopt -u nullglob

diagnosis_replay_status="not_run"
if (( ${#diagnosis_dump_files[@]} > 0 )); then
    echo "Replaying the first saved GroupNorm tensor."
    set +e
    HIP_VISIBLE_DEVICES=0 \
    TORCH_SHOW_CPP_STACKTRACES=1 \
    python "${diagnosis_root}/replay_groupnorm.py" \
        "${diagnosis_tensor_dir}" \
        2>&1 | tee "${diagnosis_artifact_dir}/probe/replay.log"
    diagnosis_replay_status="${PIPESTATUS[0]}"
    set -e
else
    echo "The probe did not create a tensor dump." \
        | tee "${diagnosis_artifact_dir}/probe/replay.log"
fi

{
    printf "run_id=%s\n" "${diagnosis_run_id}"
    printf "artifact_dir=%s\n" "${diagnosis_artifact_dir}"
    printf "probe_exit_status=%s\n" "${diagnosis_probe_status}"
    printf "replay_exit_status=%s\n" "${diagnosis_replay_status}"
    printf "tensor_dump_count=%s\n" "${#diagnosis_dump_files[@]}"
} | tee "${diagnosis_artifact_dir}/summary.txt"

echo "Diagnosis run complete: ${diagnosis_artifact_dir}"
