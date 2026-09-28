# SeaCache forward protocol

SeaCache currently supports the Cosmos3 pipelines, including the inherited Edge transformer implementation. Enable it with `cache_backend="sea_cache"`. The pipeline supplies the scheduler step, exact sigma and total step count through callbacks, and names each CFG branch through `hook.cache_context(name)`.

## Model contract

A transformer implements `SupportsSeaCache` from `vllm_omni.diffusion.cache.seacache.protocol`. The protocol extends `SupportsDecomposedForward` with `get_seacache_inputs(ctx) -> SeaCacheInputs`; it does not require TeaCache coefficients or defaults.

The uncached forward and SeaCache use the same model methods:

1. `preprocess(..., skip_modulated_input=True)` creates the packed execution tensor and model-owned intermediates.
2. `run_transformer_blocks(ctx)` returns the state after block execution. It may replace the state or mutate its hidden tensor.
3. `postprocess(ctx)` produces the public model output. Its required intermediates must already exist after preprocess, because cache hits skip the blocks.

`get_seacache_inputs` returns references to the original five-dimensional BCTHW vision latents and an optional noisy-frame mask. Cosmos3 supplies controls followed by the target, preserving the existing indicator order. The packed execution tensor may also contain action and sound tokens; it is distinct from the indicator inputs. `modulated_input=None` does not bypass SeaCache: that field belongs to the TeaCache decision contract.

SeaCache no longer accepts extractor callbacks or reads `CacheContext.extra_states`. Hook installation rejects models without the protocol and subclasses whose forward overrides the inherited decomposition.

## Residual ownership and synchronization

On a compute step, the hook snapshots the execution input before running blocks. It records `output - input_snapshot` before postprocessing, so block mutation and output conversion cannot corrupt the residual. Cosmos3's execution boundary includes the final GEN norm. A cache hit adds the extrapolated residual to the current packed input and calls postprocess.

The hook owns residual history and branch state. Existing parameter-sharding, sequence-parallel and distributed-offload decision reductions remain in the hook. Offload groups are discovered through the model's declared block attributes using the shared offloader helper.

Calls with autograd enabled, missing branch context, missing scheduler metadata or conditioning-only vision retain the existing uncached fallbacks. Invalid indicator shapes retain the existing full-compute behavior.

## Scope and validation

The protocol migration preserves the current indicator and synchronization algorithm. Before merge and Transfer quality validation, reconcile [#7939](https://github.com/vllm-project/vllm-omni/pull/7939): target-only indicators, coordinated branch-history resets, pipeline evaluation callbacks and distributed eligibility decisions. The migration does not include that fix. Its control-order tests must change when the fix is integrated; controls remain in model execution but leave the indicator.

The image implementation proposed in [#6975](https://github.com/vllm-project/vllm-omni/pull/6975) uses a two-dimensional token grid, a leading grid-token count and model-derived sigma. Those fields are not added without an image-model consumer; extending SeaCache to those models requires reconciling their indicator representation and metadata source.

CPU tests cover branch separation, reset, fallbacks, protocol rejection, inherited methods, residual mutation, offload group discovery and the Cosmos3 execution boundary. GPU output quality, input-snapshot cost and distributed GPU compatibility require separate measurements; CPU checks do not establish those results.
