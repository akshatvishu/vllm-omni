#!/usr/bin/env bash
# ROCm perf flags for VoxCPM2 on single MI300X.
# Source this inside the ROCm environment.
#
# Usage:
#   source voxcpm2_rocm_env_flags.sh set
#   source voxcpm2_rocm_env_flags.sh unset
#   source voxcpm2_rocm_env_flags.sh print

voxcpm2_rocm_flags_set() {
  export VLLM_ROCM_USE_AITER=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_LOGGING_LEVEL=INFO
  export GPU_ARCHS=gfx942
  export PYTORCH_ROCM_ARCH=gfx942
  export MIOPEN_FIND_MODE=FAST
  export MIOPE_CONV_IMMED_MODE=0
  export HIP_VISIBLE_DEVICES=0
  export HIP_CUDA_GRAPH_OPTIM_LEVEL=1
  export HIP_LAUNCH_BLOCKING=0
}

voxcpm2_rocm_flags_unset() {
  unset VLLM_ROCM_USE_AITER
  unset VLLM_WORKER_MULTIPROC_METHOD
  unset VLLM_LOGGING_LEVEL
  unset GPU_ARCHS
  unset PYTORCH_ROCM_ARCH
  unset MIOPEN_FIND_MODE
  unset MIOPE_CONV_IMMED_MODE
  unset HIP_VISIBLE_DEVICES
  unset HIP_CUDA_GRAPH_OPTIM_LEVEL
  unset HIP_LAUNCH_BLOCKING
}

voxcpm2_rocm_flags_print() {
  printf 'VLLM_ROCM_USE_AITER=%s\n' "${VLLM_ROCM_USE_AITER-}"
  printf 'VLLM_WORKER_MULTIPROC_METHOD=%s\n' "${VLLM_WORKER_MULTIPROC_METHOD-}"
  printf 'VLLM_LOGGING_LEVEL=%s\n' "${VLLM_LOGGING_LEVEL-}"
  printf 'GPU_ARCHS=%s\n' "${GPU_ARCHS-}"
  printf 'PYTORCH_ROCM_ARCH=%s\n' "${PYTORCH_ROCM_ARCH-}"
  printf 'MIOPEN_FIND_MODE=%s\n' "${MIOPEN_FIND_MODE-}"
  printf 'MIOPE_CONV_IMMED_MODE=%s\n' "${MIOPE_CONV_IMMED_MODE-}"
  printf 'HIP_VISIBLE_DEVICES=%s\n' "${HIP_VISIBLE_DEVICES-}"
  printf 'HIP_CUDA_GRAPH_OPTIM_LEVEL=%s\n' "${HIP_CUDA_GRAPH_OPTIM_LEVEL-}"
  printf 'HIP_LAUNCH_BLOCKING=%s\n' "${HIP_LAUNCH_BLOCKING-}"
}

case "${1:-set}" in
  set)
    voxcpm2_rocm_flags_set
    ;;
  unset)
    voxcpm2_rocm_flags_unset
    ;;
  print)
    voxcpm2_rocm_flags_print
    ;;
  *)
    echo "usage: source voxcpm2_rocm_env_flags.sh [set|unset|print]" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
