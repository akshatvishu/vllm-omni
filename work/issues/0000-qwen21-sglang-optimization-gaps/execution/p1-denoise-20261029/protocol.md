# P1 complete denoising comparison, 29 October 2026

Measure whether the previous approximately 2.5% warmed decode-step benefit survives a complete 40-step denoising loop on the authorized single MI300X. This is one bounded experiment, followed by a report and stop.

## Frozen implementation and workload

Use the exact two production source files from the passing fallback repair commit `d5725ef7c720d81970c3111a17b4b42f4e581d80`. Build an independent experiment branch from main `e6a7e63d04e54e28b8d35ab7fede499579ec10fc`. Preserve local main and unrelated files.

The baseline selects `retain_attention_buffers=False`, which uses the pinned main prefix-only graph buffers and native concatenation. P1 selects retained full K/V buffers. The fallback repair is shared; successful graph decode returns before that repair, so it changes neither arm's graph arithmetic. No new kernel, precision setting, block-size tuning or library modification is part of this experiment.

Use actual 32-block Qwen-Image-2.1 weights at snapshot `d26bb61231c349cf6b7896fa83353113880e1ba3`. Reuse the saved four-reference context with SHA256 `a26ac7fcca255299c23adac5f193c6512832a326e916815db042628f9dfe3500`. Target length is 3072 tokens. Use BF16, CFG 4, 40 saved timesteps, TP1/SP1, TORCH_SDPA, dynamic regional compilation and disabled Inductor cudagraphs on gfx942. Reuse the preceding compilation cache. The saved encoded inputs and initial latents are unchanged. No artificial prefix perturbations are used.

Run the production `QwenImage21CFGParallelMixin.diffuse()` method and inherited CFG/scheduler methods. A lightweight holder supplies the already-loaded transformer, actual scheduler and saved context. It does not instantiate the encoder or VAE. Disable only the progress display.

## Correctness gate

Run native, P1, then native again before timing. These runs also warm compilation. Save all 80 branch target predictions and all 40 updated latents per run. Require byte equality, finite final outputs and matching final metadata. Compare the 128 prefix tensors across the independent runs. Require both CFG graphs to capture once and the expected retained-buffer policy. Verify dead request owners and entries are released after each run. Stop on any failure; do not attribute a native A/A mismatch to P1.

During timing, install no output or scheduler observers. Compare each final latent against the native reference after the timing interval. Require both graphs captured once, correct buffer policy, empty managers after cleanup and no new Dynamo graph compilation in timed runs. Save final tensors remotely for independent CPU audit.

## Timing and evaluation

Predeclare two native/native control pairs, one before and one after six alternating native/P1 pairs. Do not expand the sample budget based on the answer. Each invocation starts a fresh request with empty graph-manager state. Prefill, graph buffer allocation, prefix transfer, capture/warmup, all 39 decode steps, CFG combinations and scheduler updates are inside the interval. Include the production input construction and output clone operations.

The primary metric is synchronized wall time around `diffuse()`. Also record CUDA/HIP event elapsed time over the same interval. Event elapsed time includes host-induced GPU gaps and capture setup; it is not the sum of kernel durations. Scheduler creation, timestep preparation, initial-latent cloning, model loading/compilation warmup, prior-request cleanup, artifact writing and validation are outside timing. The normal production geometry release inside `diffuse()` remains included.

Use the median paired saving, median paired percentage, positive pair count and maximum absolute native/native difference. Keep the denoising speed hypothesis only if all correctness checks pass, at least five of six pairs improve, median paired saving exceeds the largest observed A/A difference, and median paired improvement is at least 1%. This is a bounded screening rule, not a statistical confidence interval. Retain negative and inconclusive results.

The maximum worker runtime is 1200 seconds. No profiler or extra sweep is planned. This experiment measures one complete denoising workload in one process. It does not qualify HTTP request latency, image quality, encoder/VAE repeatability, different edit shapes, NVIDIA, concurrency, cancellation, automatic cleanup or peak-memory capacity. The earlier 3 GiB retained-storage cost remains relevant but is not a new peak-memory claim.

Raw tensors and logs stay on the authorized host. Save the findings and provenance under the canonical local issue directory. Restore remote main, verify GPU idle and release only this controller's lock. Report and stop before any next experiment.
