# Hunyuan Image 3.0 AITER GroupNorm diagnosis

Run every command from the vLLM Omni repository root on the ROCm machine. The commands use plain `python` because the SSH environment already provides the required interpreter.

Every generated log, image, tensor dump, patch record, and comparison result is written under `hunyuan_groupnorm_diagnosis/artifacts`.

## Run the complete diagnosis

The runner performs the bad revision run, fixed revision run, image comparison, live GroupNorm probe, and tensor replay:

```bash
./hunyuan_groupnorm_diagnosis/run_diagnosis.sh
```

The default baseline is `tests/assets/hunyuan/hunyuan_baseline.png`. Pass a known good ROCm image and an optional run ID when available:

```bash
./hunyuan_groupnorm_diagnosis/run_diagnosis.sh \
  hunyuan_groupnorm_diagnosis/artifacts/baseline/known_good.png \
  mi325-known-good
```

Set the four physical GPU IDs through `HUNYUAN_GROUPNORM_DEVICES`:

```bash
HUNYUAN_GROUPNORM_DEVICES=4,5,6,7 \
  ./hunyuan_groupnorm_diagnosis/run_diagnosis.sh
```

Each invocation creates `hunyuan_groupnorm_diagnosis/artifacts/runs/<run-id>`. The runner never deletes or overwrites an earlier run.

## What the test must prove

GroupNorm is the cause only if both checks pass.

1. The bad revision fails image accuracy and the GroupNorm fix restores image accuracy.
2. AITER and PyTorch differ on the same live tensor used by Hunyuan GroupNorm.

The existing `aiter_grp_log` and `no_aiter_grp_log` cannot confirm the cause because one run uses 50 steps and the other uses 1 step.

Do not use `VLLM_ROCM_USE_AITER=0` as a GroupNorm control. The GroupNorm patch calls `is_aiter_found_and_supported()`, which does not read that flag.

## Record the environment

```bash
set -o pipefail
{
  python - <<'PY'
import aiter
import torch

print("torch", torch.__version__)
print("hip", torch.version.hip)
print("aiter", aiter.__file__)
print("device_count", torch.cuda.device_count())
PY
  printf "vllm_omni "
  git rev-parse HEAD
  printf "aiter "
  git -C ../aiter rev-parse HEAD
} 2>&1 | tee hunyuan_groupnorm_diagnosis/artifacts/environment.txt
```

The environment record must show a nonempty HIP version and the intended AITER installation.

## Create isolated source copies

Commit `422d8fd8f` contains the bad AITER GroupNorm path. Commit `b01721535` changes runtime behavior only in `patch_groupnorm.py`. The other changed files are tests.

Save the revision scope:

```bash
git diff --name-only 422d8fd8f..b01721535 \
  | tee hunyuan_groupnorm_diagnosis/artifacts/revision_files.txt

git diff 422d8fd8f..b01721535 -- \
  vllm_omni/platforms/rocm/patch/worker/patch_groupnorm.py \
  | tee hunyuan_groupnorm_diagnosis/artifacts/groupnorm_runtime.diff
```

Create two shared clones inside the diagnosis folder:

```bash
set -e
test ! -e hunyuan_groupnorm_diagnosis/worktrees/bad
test ! -e hunyuan_groupnorm_diagnosis/worktrees/fixed

{
  git clone --shared --no-checkout . \
    hunyuan_groupnorm_diagnosis/worktrees/bad
  git -C hunyuan_groupnorm_diagnosis/worktrees/bad \
    checkout --detach 422d8fd8f

  git clone --shared --no-checkout . \
    hunyuan_groupnorm_diagnosis/worktrees/fixed
  git -C hunyuan_groupnorm_diagnosis/worktrees/fixed \
    checkout --detach b01721535
} 2>&1 | tee hunyuan_groupnorm_diagnosis/artifacts/source_setup.log
```

`PYTHONPATH` in the following commands makes each run load code from the selected clone. Both runs still use the same Python, PyTorch, AITER, model cache, and GPU environment.

## Run the bad revision

```bash
set -o pipefail
test ! -e hunyuan_groupnorm_diagnosis/artifacts/bad/images

HIP_VISIBLE_DEVICES=0,1,2,3 \
VLLM_ROCM_USE_AITER=1 \
VLLM_LOGGING_LEVEL=INFO \
PYTHONPATH=hunyuan_groupnorm_diagnosis/worktrees/bad \
python \
  hunyuan_groupnorm_diagnosis/worktrees/bad/examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality text2img \
  --deploy-config hunyuan_groupnorm_diagnosis/worktrees/bad/vllm_omni/deploy/hunyuan_image3_dit.yaml \
  --prompts "A brown and white dog is running on the grass." \
  --output hunyuan_groupnorm_diagnosis/artifacts/bad/images \
  --steps 50 \
  --guidance-scale 2.5 \
  --seed 42 \
  --height 1024 \
  --width 1024 \
  --bot-task none \
  --sys-type en_unified \
  --enforce-eager \
  2>&1 | tee hunyuan_groupnorm_diagnosis/artifacts/bad/run.log
```

## Run the fixed revision

Keep every model argument and environment setting unchanged.

```bash
set -o pipefail
test ! -e hunyuan_groupnorm_diagnosis/artifacts/fixed/images

HIP_VISIBLE_DEVICES=0,1,2,3 \
VLLM_ROCM_USE_AITER=1 \
VLLM_LOGGING_LEVEL=INFO \
PYTHONPATH=hunyuan_groupnorm_diagnosis/worktrees/fixed \
python \
  hunyuan_groupnorm_diagnosis/worktrees/fixed/examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality text2img \
  --deploy-config hunyuan_groupnorm_diagnosis/worktrees/fixed/vllm_omni/deploy/hunyuan_image3_dit.yaml \
  --prompts "A brown and white dog is running on the grass." \
  --output hunyuan_groupnorm_diagnosis/artifacts/fixed/images \
  --steps 50 \
  --guidance-scale 2.5 \
  --seed 42 \
  --height 1024 \
  --width 1024 \
  --bot-task none \
  --sys-type en_unified \
  --enforce-eager \
  2>&1 | tee hunyuan_groupnorm_diagnosis/artifacts/fixed/run.log
```

## Compare the images

Use the image from the last known good ROCm run as the baseline. Save that image as `hunyuan_groupnorm_diagnosis/artifacts/baseline/known_good.png`.

If no known good ROCm image exists, use `tests/assets/hunyuan/hunyuan_baseline.png` for the first check. A result against that baseline is preliminary because the current accuracy test does not list ROCm hardware.

Run the comparison:

```bash
set -o pipefail
MPLCONFIGDIR=hunyuan_groupnorm_diagnosis/artifacts/matplotlib_cache \
python hunyuan_groupnorm_diagnosis/compare_images.py \
  --baseline hunyuan_groupnorm_diagnosis/artifacts/baseline/known_good.png \
  2>&1 | tee hunyuan_groupnorm_diagnosis/artifacts/image_comparison.txt
```

Use this command instead when only the checked in baseline is available:

```bash
set -o pipefail
MPLCONFIGDIR=hunyuan_groupnorm_diagnosis/artifacts/matplotlib_cache \
python hunyuan_groupnorm_diagnosis/compare_images.py \
  --baseline tests/assets/hunyuan/hunyuan_baseline.png \
  2>&1 | tee hunyuan_groupnorm_diagnosis/artifacts/image_comparison.txt
```

The repository thresholds are mean error at most `0.03`, p99 error at most `0.3`, SSIM at least `0.97`, and PSNR at least `30`.

| Bad revision | Fixed revision | Conclusion |
| --- | --- | --- |
| Fails | Passes | The GroupNorm runtime change causes the image regression. |
| Fails | Fails in the same way | GroupNorm is not confirmed. |
| Passes | Passes | The reported regression was not reproduced. |
| No known good ROCm baseline | Outputs differ | The image test is inconclusive. Obtain a known good ROCm result. |

## Find the first wrong GroupNorm call

Apply the included diagnostic patch to the bad clone. The patch runs PyTorch and AITER on the same live GroupNorm input. It stops on the first output or dtype mismatch and saves the input, weight, bias, expected output, actual output, group count, epsilon, and autocast state.

```bash
set -e
{
  git -C hunyuan_groupnorm_diagnosis/worktrees/bad apply --check \
    ../../diagnostic_groupnorm.patch
  git -C hunyuan_groupnorm_diagnosis/worktrees/bad apply \
    ../../diagnostic_groupnorm.patch
  git -C hunyuan_groupnorm_diagnosis/worktrees/bad diff -- \
    vllm_omni/platforms/rocm/patch/worker/patch_groupnorm.py
} 2>&1 | tee hunyuan_groupnorm_diagnosis/artifacts/probe/applied_patch.log
```

Run one Hunyuan step. One step is enough because the probe runs during VAE decode.

```bash
set -o pipefail
set +e

HIP_VISIBLE_DEVICES=0,1,2,3 \
VLLM_ROCM_USE_AITER=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
HUNYUAN_GROUPNORM_DIAG_DIR=hunyuan_groupnorm_diagnosis/artifacts/tensor_dumps \
PYTHONPATH=hunyuan_groupnorm_diagnosis/worktrees/bad \
python \
  hunyuan_groupnorm_diagnosis/worktrees/bad/examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality text2img \
  --deploy-config hunyuan_groupnorm_diagnosis/worktrees/bad/vllm_omni/deploy/hunyuan_image3_dit.yaml \
  --prompts "A brown and white dog is running on the grass." \
  --output hunyuan_groupnorm_diagnosis/artifacts/probe/images \
  --steps 1 \
  --guidance-scale 2.5 \
  --seed 42 \
  --height 1024 \
  --width 1024 \
  --bot-task none \
  --sys-type en_unified \
  --enforce-eager \
  2>&1 | tee hunyuan_groupnorm_diagnosis/artifacts/probe/run.log

groupnorm_probe_status=${PIPESTATUS[0]}
printf "probe_exit_status=%s\n" "${groupnorm_probe_status}" \
  | tee hunyuan_groupnorm_diagnosis/artifacts/probe/status.txt
set -e
```

A nonzero exit is expected when the probe finds a mismatch. The log line beginning with `GROUPNORM_PROBE` records the live tensor shape and dtypes. The exception records the output dtypes and numerical error.

## Replay the saved tensor

Replay the first saved tensor through PyTorch and AITER without loading Hunyuan:

```bash
set -o pipefail
HIP_VISIBLE_DEVICES=0 \
TORCH_SHOW_CPP_STACKTRACES=1 \
python hunyuan_groupnorm_diagnosis/replay_groupnorm.py \
  hunyuan_groupnorm_diagnosis/artifacts/tensor_dumps \
  2>&1 | tee hunyuan_groupnorm_diagnosis/artifacts/probe/replay.log
```

Use this decision rule:

| Result | Conclusion |
| --- | --- |
| Bad image fails, fixed image passes, and the live probe differs | GroupNorm is confirmed as the regression cause. |
| The live AITER and PyTorch outputs match | The proposed GroupNorm numerical cause is false in that environment. |
| AITER raises on the live input | AITER does not support the input contract. Keep the saved dump and stack trace. |
| Only the output dtype differs | Autocast behavior differs. The image comparison determines whether it causes the regression. |

## Fix ownership after confirmation

If the bad and fixed image runs plus the live probe confirm the failure, vLLM Omni owns the functional fix because it automatically replaced PyTorch GroupNorm.

If the replay shows that AITER accepts mixed dtypes and reads them incorrectly, AITER also owns a safety fix. AITER must reject the mixed dtype call or support it explicitly.
