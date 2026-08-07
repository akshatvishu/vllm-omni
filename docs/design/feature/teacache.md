# TeaCache

This document describes the native TeaCache integration used by migrated diffusion models. It does not establish a speedup, quality, or memory result for any model. Those results require a model-specific benchmark and an uncached comparison.

## Overview

TeaCache measures a timestep-derived tensor between diffusion steps. When the rescaled accumulated distance is below the configured threshold, it reuses the detached residual from a declared transformer block region:

```text
boundary input -> transformer blocks -> boundary output
                    residual = output - input
cache hit: current boundary input + residual
```

The residual is not a final diffusion prediction. The boundary must be placed after model-specific preprocessing and before any output projection, scheduler operation, or postprocessing that changes the cached tensor's shape.

## Native interface

The interface is defined in `vllm_omni/diffusion/cache/teacache/interface.py`.

Models that support native TeaCache expose:

- `supports_teacache = True`
- `tea_cache_model_key`, a stable model key
- `tea_cache_executor`, initially `None` and populated by `TeaCacheBackend`
- `get_teacache_coefficients()`

The model's forward method owns the boundary. It builds a `compute_fn` for the block region and calls `TeaCacheRuntime.run` with the timestep metric and boundary input tuple. The runtime validates residual arity and shape, snapshots inputs before execution, isolates CFG branches, and resets between generations.

`TeaCacheBackend` installs `TeaCacheRuntime` on a native `pipeline.transformer`. Models that still use the old extractor and hook protocol remain on that legacy path until they are migrated. Hunyuan Image 3 is not a custom backend enabler.

## Hunyuan Image 3 boundary

The Hunyuan boundary is the `self.layers` loop in `HunyuanImage3Model.forward`.

```text
Hunyuan preprocessing and SP padding
    -> TeaCacheRuntime(self.layers)
    -> SP output gather and model output
```

The pipeline already computes `t_emb = self.time_embed(timestep)` before `patch_embed`. The same tensor is passed to `HunyuanImage3Model` as `tea_cache_modulated_input`, so the cache metric does not use rank-local hidden states. The pipeline passes `tea_cache_do_true_cfg=True` only for CFG-parallel execution. Sequential CFG uses one complete boundary tensor and keeps this flag false.

Hunyuan skips TeaCache on the first image step. That step transitions from prompt and image placeholder inputs to the stable image-only denoising shape. The model also bypasses TeaCache for text generation, unconditional CFG prefill, attention or hidden-state collection, `use_cache=True`, and a boundary shape or metric shape change.

Hunyuan step execution continues to reject diffusion cache backends because its runtime state is not request-scoped. This restriction must remain until request lifecycle ownership is implemented.

## Configuration and coefficients

`TeaCacheConfig` requires five finite polynomial coefficients and a positive finite threshold. The backend resolves user-provided coefficients first and otherwise calls the model getter. Coefficients must be collected at the exact native boundary. A tuple from an older boundary is not evidence for the new boundary.

For Hunyuan Image 3, the current tuple remains a provisional finite default so the backend can be exercised. It is not a fitted result for the native decoder-layer boundary and must not be used as an image-quality or speed claim.

The native coefficient collector in `coefficient_estimator.py` records the metric and first computed boundary output through `DataCollectionExecutor`. A collection run should record the model revision, prompt, image size, seed, timestep schedule, guidance mode, parallelism, dtype, boundary shape, and resulting coefficient tuple.

## CPU validation

The CPU tests use a tiny Hunyuan configuration and replace its decoder layers with deterministic counting layers. They prove first-step bypass, later miss, later hit, metric miss, residual reuse, changed-shape recomputation, unsupported modes, explicit `use_cache` behavior, pipeline metric handoff, and backend refresh.

Run them with the repository's core model level:

```bash
./.venv/bin/pytest --run-level core_model -q \
  tests/diffusion/cache/test_teacache_unit.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_teacache.py
```

The generic runtime tests also cover empty boundaries, arity mismatch, output-shape mismatch, nonfinite metrics, CFG branch isolation, and lifecycle reset.

## Distributed validation

Sequence-parallel validation must run on two ranks after the CPU tests pass. It should assert that both ranks make the same cache decision, that the metric is identical across ranks, that local residuals preserve rank sharding, and that the postprocessor restores the original sequence shape. Do not combine sequence and CFG parallelism in the first distributed test.

A separate two-rank CFG-parallel test must verify positive and negative residual isolation with `do_true_cfg=True`. Sequential CFG must use `do_true_cfg=False`.

## Real-model validation

Use `--run-level advanced_model` for real-weight tests. Run uncached and cached generation with the same model, prompt, seed, image size, guidance scale, dtype, and number of steps. Compare outputs with the repository's existing image metrics and record computed layer passes, elapsed time, and memory. Do not copy speed, quality, or threshold values from the TeaCache paper to Hunyuan Image 3.

## Migration checklist

1. Read the model's upstream and local forward path before choosing the boundary.
2. Add the native capability attributes and remove the model's old TeaCache protocol stubs.
3. Wrap only the transformer block region in a local `compute_fn`.
4. Pass a timestep metric that is stable across ranks and explicit CFG mode.
5. Add negative CPU tests before running real-weight tests.
6. Collect coefficients at the new boundary.
7. Run distributed, quality, latency, and memory validation separately.
