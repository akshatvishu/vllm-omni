# TeaCache

TeaCache reuses transformer block residuals across denoising steps. This document explains the cache decision, state ownership and model interface.

---

## Table of Contents

- [Overview](#overview)
- [Definitions](#definitions)
- [Cache Decision](#cache-decision)
- [Architecture](#architecture)
- [Adding TeaCache to a Model](#adding-teacache-to-a-model)
- [Legacy Extractor Path](#legacy-extractor-path)
- [Coefficient Estimation](#coefficient-estimation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Reference Implementations](#reference-implementations)

## Overview

See the [TeaCache paper](https://arxiv.org/abs/2411.19108) for the method and the [reference implementation](https://github.com/ali-vilab/TeaCache) for the authors' code. See the [user guide](../../user_guide/diffusion/cache_acceleration/teacache.md) for configuration, measurements and model restrictions.

In Sections 3.2 and 3.3, the authors use changes in the timestep-modulated noisy input to estimate changes in transformer output. A fitted polynomial rescales the input distance before accumulation. The accumulated value is an estimate used to choose when to compute, not a bound on output error. Section 3.4 specifies that the cached value is the transformer output minus its input, so a cache hit adds the stored residual to the current input.

A model supports TeaCache by splitting its `forward` into `preprocess`, `run_transformer_blocks` and `postprocess`. The TeaCache hook calls the same methods, so cached and uncached execution share one implementation. Older models use extractor functions instead, which copy the model's forward and can drift from it. The hook still accepts extractors for models that have not been ported; see [Legacy Extractor Path](#legacy-extractor-path).

`preprocess` can return `modulated_input=None` when a call must run without TeaCache. The hook then runs the blocks and postprocess without advancing cache state, and the coefficient collector skips that call.

## Definitions

- **`modulated_input`** is the model-provided tensor used to decide whether to reuse a residual. Qwen-Image uses its first block's normalized and modulated image input; other models can provide a different tensor.
- **Cached block range** is the computation in `run_transformer_blocks` that a cache hit skips.
- **Residual** is the difference between the cached block range's output and its input. A cache hit adds the stored residual to the current input.
- **Accumulated distance** is the running sum of absolute polynomial-rescaled changes in `modulated_input`. Reaching the threshold resets the sum to zero.
- **Cache state** holds a branch's previous `modulated_input`, residual, accumulated distance and count of eligible calls.

## Cache Decision

The first eligible call and the configured warmup calls always run the blocks. For later calls, the hook compares `modulated_input` with its value from the previous eligible call:

```text
distance = mean(abs(current - previous)) / (mean(abs(previous)) + 1e-8)
accumulated_distance += abs(polynomial(distance))
```

If the accumulated distance is below `rel_l1_thresh` and a residual is available, the hook reuses it. Otherwise, it runs the blocks. Reaching the threshold resets the accumulated distance to zero. The hook saves the current `modulated_input` after both compute and cache-hit calls.

!!! note "Implementation details"
    The formula above describes `TeaCacheHook`. The paper's Equations 4 and 7 use an L1 norm ratio and a sum of polynomial estimates. Our hook uses means, adds `1e-8` to the denominator, and takes the absolute value of each polynomial estimate. Without the epsilon, the ratio of means equals the ratio of sums for tensors of the same size. Warmup counts, CFG state selection and sequence-parallel synchronization are also details of this implementation.

### Worked Example

Consider one cache state with `rel_l1_thresh=0.2` and one warmup call. For illustration, use `polynomial(distance) = 2 * distance`. These numbers explain the decision rule; they are not fitted coefficients for a model.

| Eligible call | Relative L1 distance | Rescaled distance | Accumulated distance before decision | Action |
| --- | ---: | ---: | ---: | --- |
| 0 | Not compared | Not computed | 0 | Warmup: run blocks and store a residual. |
| 1 | 0.03 | 0.06 | 0.06 | Reuse the residual. |
| 2 | 0.04 | 0.08 | 0.14 | Reuse the residual. |
| 3 | 0.05 | 0.10 | 0.24 | Run blocks, replace the residual and reset the total to 0. |
| 4 | 0.01 | 0.02 | 0.02 | Reuse the new residual. |

!!! note "CFG branches and bypassed calls"
    With sequential classifier-free guidance (CFG), `do_true_cfg=True` makes the hook alternate between positive and negative cache states. Each state has its own warmup count and accumulated distance. A protocol call with `modulated_input=None` runs uncached without advancing that alternation or changing either state's residual.

For CFG parallelism, the hook selects state by rank. A rank owning several independently cached branches cannot distinguish them by rank alone; SenseNova rejects the affected three-branch editing configuration. Bagel packs its active CFG branches into one forward and uses one cache state for the packed input.

Sequence-parallel ranks average the L1 numerator and denominator before division so they make the same decision. TeaCache does not add general synchronization across weight-sharding or offload groups; each model must restrict combinations that need it.

## Architecture

The TeaCache system consists of these components:

| Component | Purpose | Location |
| ----------- | --------- | ---------- |
| `SupportsTeaCache`, `ForwardState`, `TeaCacheDefaults` | Decomposed-forward protocol, the state passed between its methods, and model defaults | `vllm_omni/diffusion/cache/teacache/protocol.py` |
| `TeaCacheHook` | Decides whether to run or skip the transformer blocks and stores residuals | `vllm_omni/diffusion/cache/teacache/hook.py` |
| [`CacheContext`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/#vllm_omni.diffusion.cache.CacheContext) | Legacy: dataclass an extractor returns | `vllm_omni/diffusion/cache/teacache/context.py` |
| [`EXTRACTOR_REGISTRY`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/teacache/extractors/#vllm_omni.diffusion.cache.teacache.extractors.EXTRACTOR_REGISTRY) | Legacy: maps transformer class names to extractor functions | `vllm_omni/diffusion/cache/teacache/extractors.py` |
| [`TeaCacheConfig`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/#vllm_omni.diffusion.cache.TeaCacheConfig) | Configuration including thresholds and polynomial coefficients | `vllm_omni/diffusion/cache/teacache/config.py` |

The hook owns:

- CFG-aware state management (separate states for positive/negative branches)
- Rank-based state selection for supported CFG-parallel configurations
- L1 distance computation with polynomial rescaling
- Residual caching and reuse

---

## Adding TeaCache to a Model

Implement `SupportsTeaCache` from `vllm_omni.diffusion.cache.teacache.protocol` on the transformer. The reference is `QwenImageTransformer2DModel` in `vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py`.

| Method | Responsibility |
| --- | --- |
| `preprocess(..., skip_modulated_input=...)` | Prepare embeddings, masks and block inputs in a `ForwardState`. Compute `modulated_input` unless skipped or the call is ineligible for caching. |
| `run_transformer_blocks(ctx)` | Run the cached block range and return the updated state. |
| `postprocess(ctx)` | Convert the state into the model's public output. It also runs on cache hits. |
| `get_teacache_defaults()` | Return coefficients, the default threshold and the optional warmup count in `TeaCacheDefaults`. |

Use a model-specific dataclass for `ForwardState.intermediates`. Preprocessing must provide everything postprocessing needs, including output options such as `return_dict`, because cache hits skip the blocks. `encoder_hidden_states` and `temb` may be `None` when the model does not need them.

Keep the public forward's explicit arguments because other hooks inspect its signature. Qwen-Image calls the shared methods as follows:

??? code "Qwen-Image forward"

    ```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        guidance: torch.Tensor = None,
        attention_kwargs: dict[str, Any] | None = None,
        additional_t_cond: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor, ...]:
        ctx = self.preprocess(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            timestep,
            img_shapes,
            txt_seq_lens,
            guidance,
            attention_kwargs,
            additional_t_cond,
            return_dict,
            skip_modulated_input=True,
        )
        ctx = self.run_transformer_blocks(ctx)
        return self.postprocess(ctx)
    ```

The uncached forward passes `skip_modulated_input=True` to avoid computing an unused `modulated_input`. Hook installation rejects a subclass that overrides `forward` while inheriting `preprocess` from a parent class, because caching would bypass the override.

To estimate coefficients for a new model, see [Coefficient Estimation](#coefficient-estimation).

### Model Notes

#### Cosmos3

Cosmos3 uses the first GEN layer's `input_layernorm` output as `modulated_input`. The cached range includes the final GEN norm. The coefficients came from a range ending before that norm. Changing `modulated_input` or the cached range requires a new quality comparison; refit the coefficients if it fails.

TeaCache rejects HSDP, layerwise offload and distributed layerwise offload for Cosmos3 for the following reasons:

- Computing `modulated_input` reads the first block's weights before its forward. Those weights may still be sharded or unavailable through the offload hooks.
- Layerwise prefetch runs asynchronously, and the early read has not been validated against that prefetch.
- Ranks sharing sharded weights must agree on whether to execute the blocks. TeaCache currently synchronizes decisions only across sequence-parallel ranks.

Transfer requests run uncached because their CFG branches use different GEN layouts.

#### Bagel

Bagel packs active CFG branches into one forward, with separate rows in the cached residual. It preserves prefix KV caches and applies CFG after the cached blocks. Calls outside the CFG interval use a different layout and run uncached. The SP and CFG-parallel generation paths call `forward_single_branch`, which the hook does not wrap.

The former extractor ignored CFG arguments and ran only the conditional branch. Measurements from that path describe a different workload from the corrected CFG implementation.

---

## Legacy Extractor Path

MiniMax-H3 still uses an extractor registered in `EXTRACTOR_REGISTRY`. The extractor returns a `CacheContext` containing the block inputs and callbacks for block execution and postprocessing. See `extract_minimax_h3_context` in [extractors.py](../../../vllm_omni/diffusion/cache/teacache/extractors.py) for a working example. New model integrations should use `SupportsTeaCache` so cached and uncached execution share the model's methods.

## Coefficient Estimation

The polynomial maps changes in `modulated_input` to changes in block output. Coefficients are specific to the model, the tensor used as `modulated_input` and the cached block range. Values borrowed from another model require validation before they can serve as defaults.

The authors fitted their polynomials using 70 prompts from T2V-CompBench, with 10 prompts for each of seven attributes (Section 4.1). That sample count describes their experiment. Choose calibration prompts that cover the target workload and validate the fit on separate prompts. Their polynomial-order ablation found that gains saturated at degree four (Section 4.3 and Table 3).

[`DataCollectionHook`](../../../vllm_omni/diffusion/cache/teacache/coefficient_estimator.py) always runs the cached block range and records `modulated_input` together with `hidden_states` before postprocessing. It uses `SupportsDecomposedForward`, so collection does not require existing TeaCache defaults. Calls with `modulated_input=None` run normally but are excluded from the collected data.

`TeaCacheCoefficientEstimator` uses an adapter to load the pipeline, select its transformer and install the collector. `_MODEL_ADAPTERS` currently contains adapters for Bagel, Flux2, LongCat and Stable Audio. Use `DefaultAdapter` and those implementations as references when adding an adapter.

For a registered adapter:

1. Create `TeaCacheCoefficientEstimator(model_path=..., model_type=...)` with the adapter's registered name.
2. Call `collect_from_prompt` with representative prompts, seeds and inference-step counts.
3. Call `estimate(poly_order=4)` to fit the five coefficients, ordered from the fourth-degree term to the constant term.
4. Compare cached and uncached generation on separate prompts and seeds before choosing defaults.

!!! note "Model initialization"
    Some models need distributed state and a vLLM forward context during collection. Initialize the contexts required by the model before running it. See `DefaultAdapter.load_pipeline` and the model-specific adapters in [coefficient_estimator.py](../../../vllm_omni/diffusion/cache/teacache/coefficient_estimator.py) for loading requirements.

## Testing

Use fake or tiny models to check the execution contract:

- The split uncached forward preserves the original arguments, outputs and return types.
- Compute calls run the intended block range; cache hits skip it and reconstruct the expected output from the residual.
- In-place block changes do not corrupt saved residuals.
- CFG branches keep their own state, and bypassed calls do not advance branch pairing.
- Refresh clears state between requests, and coefficient collection works without an extractor.

Use real weights to compare request latency, denoising time and output similarity against uncached generation on representative prompts. Keep the checkpoint, seeds and generation settings fixed. Record SSIM or LPIPS alongside saved outputs for image and video models. Check supported distributed and offload configurations separately.

!!! note "Verifying reuse"
    An initialization message confirms hook installation. For models using `TeaCacheHook`, enable `VLLM_LOGGING_LEVEL=DEBUG` and inspect `TeaCache step=... cache_hit=True` to confirm residual reuse.

See the [TeaCache user guide](../../user_guide/diffusion/cache_acceleration/teacache.md) for configuration and measured results.

## Troubleshooting

### Unknown model type

For a new model, implement all methods required by [`SupportsTeaCache`](#adding-teacache-to-a-model). For an extractor-based model, check the transformer class name and its entry in `EXTRACTOR_REGISTRY`. The registered extractor must match the model's forward behavior.

### Cannot find coefficients

Return calibrated coefficients from `get_teacache_defaults()`, or supply them explicitly through `cache_config`. Extractor-based integrations use `_MODEL_COEFFICIENTS` in `config.py`. The configuration requires five polynomial coefficients.

### Output changes are too large

Lower `rel_l1_thresh` and repeat the same cached and uncached comparison. If the differences remain unacceptable, check that the coefficients match `modulated_input` and the cached block range, then refit them or disable caching. A speedup alone does not establish output quality.

## Reference Implementations

Complete examples in the codebase:

| Model | Path | Pattern | Notes |
| ------- | ------ | --------- | ------- |
| **Qwen-Image** | `vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py` | Decomposed forward | `preprocess`, `run_transformer_blocks`, `postprocess` |
| **Bagel** | `vllm_omni/diffusion/models/bagel/bagel_transformer.py` | Decomposed forward | Packed tokens and batched CFG |
| **TeaCache Core** | `vllm_omni/diffusion/cache/teacache/` | Base implementation | Hook and config |
| **Coefficient Estimator** | `vllm_omni/diffusion/cache/teacache/coefficient_estimator.py` | Estimation tool | Adapter pattern |
