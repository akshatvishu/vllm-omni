# Environment and configuration trace

## Recommended settings

The scripts do not set a new performance environment flag. They use the checked deploy configs and their platform overrides. A flag should be added only after a failing MI300X run proves that it is required.

| Setting | Set point | Consumer | Decision |
| --- | --- | --- | --- |
| `VLLM_ROCM_USE_AITER=0` | The existing Ming Omni TTS ROCm Docker command sets it. | vLLM parses it in `/home/aja/vllm/vllm/envs.py`. vLLM ROCm platform and operator selection read `envs.VLLM_ROCM_USE_AITER`, including attention, linear, MoE, and RMSNorm paths. vLLM Omni also reads it in `vllm_omni/platforms/rocm/platform.py` when choosing RMSNorm operator priority. | The flag has end consumers. The checked upstream default is already false, so the new validation scripts omit the redundant assignment. Keep it in a final recipe only if the tested container default or AITER setup makes the explicit value useful for reproducibility. |
| `HIP_VISIBLE_DEVICES` | A user could set it in the shell. | The HIP runtime and PyTorch consume it. vLLM Omni also writes and removes it in `RocmOmniPlatform.set_device_control_env_var` and `unset_device_control_env_var`. | The scripts do not set it. Every explicit deploy config used here assigns device `0`, which vLLM Omni maps to the ROCm device control variable. The single stage auto configs run on the only visible device. |
| `MING_CFM_CUDAGRAPH` | Optional shell setting for Ming Omni TTS. | `vllm_omni/model_executor/models/ming_tts/ming_tts_llm.py` reads it while constructing the model and uses it to select or skip the inner CFM graph path. The graph path falls back to eager after an exception. | The flag has a direct consumer, but the scripts do not set it. The existing MI300X recipe records a validated command without this setting. Set it to `0` only if the MI300X log proves that the inner graph path fails or gives wrong output. |
| Qwen3 TTS `platforms.rocm.stages[1].enforce_eager` | `vllm_omni/deploy/qwen3_tts.yaml` | `load_deploy_config` loads the platform block, `_apply_platform_overrides` merges it for ROCm, and stage construction forwards `enforce_eager` to the engine. `tests/config/test_config_factory.py::test_qwen3_tts_rocm_disables_code2wav_outer_cudagraph` verifies the resolved value. | Use this checked configuration path. No environment flag is needed. |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | The Ming, Qwen3 TTS, and OmniVoice examples set it before engine imports. | Upstream vLLM multiprocessing configuration consumes it when selecting the worker start method. | Do not duplicate it in the shell scripts because the examples already set it. |
| `HF_TOKEN` | The user may provide it to the container or process environment. | Hugging Face Hub consumes it when downloading gated or private files. | Stable Audio may require it. The scripts never print its value. |

## Upstream design used

The configuration design follows the checked vLLM and vLLM Omni pattern. Hardware differences belong in a platform specific config override when they change engine behavior, while environment variables are reserved for upstream features that already have a defined consumer. The Qwen3 TTS ROCm eager override is the model for any new required ROCm setting.
