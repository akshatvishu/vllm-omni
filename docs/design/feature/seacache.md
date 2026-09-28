# SeaCache forward protocol

SeaCache currently supports the Cosmos3 pipelines, including the inherited Edge transformer implementation. Enable it with `cache_backend="sea_cache"`. The pipeline supplies the scheduler step, exact sigma and total step count through callbacks, registers the active branch tuple through `hook.begin_step(branches)` before each velocity evaluation, and names each CFG branch through `hook.cache_context(name)`.

## Model contract

A transformer implements `SupportsSeaCache` from `vllm_omni.diffusion.cache.seacache.protocol`. The protocol extends `SupportsDecomposedForward` with `get_seacache_inputs(ctx) -> SeaCacheInputs`; it does not require TeaCache coefficients or defaults.

The uncached forward and SeaCache use the same model methods:

1. `preprocess(..., skip_modulated_input=True)` creates the packed execution tensor and model-owned intermediates.
2. `run_transformer_blocks(ctx)` returns the state after block execution. It may replace the state or mutate its hidden tensor.
3. `postprocess(ctx)` produces the public model output. Its required intermediates must already exist after preprocess, because cache hits skip the blocks.

`get_seacache_inputs` returns the original five-dimensional BCTHW target tensor in `latent` and an optional noisy-frame mask. Cosmos3 supplies only the noisy target, retaining any clean prefix frames within that target. Separate control hints remain in model execution but do not enter the indicator. The packed execution tensor may also contain action and sound tokens; it is distinct from the indicator inputs. `modulated_input=None` does not bypass SeaCache: that field belongs to the TeaCache decision contract.

SeaCache no longer accepts extractor callbacks or reads `CacheContext.extra_states`. Hook installation rejects models without the protocol and subclasses whose forward overrides the inherited decomposition.

## Residual ownership and synchronization

On a compute step, the hook snapshots the execution input before running blocks. It records `output - input_snapshot` before postprocessing, so block mutation and output conversion cannot corrupt the residual. Cosmos3's execution boundary includes the final GEN norm. A cache hit adds the extrapolated residual to the current packed input and calls postprocess.

The hook owns residual history and branch state. Parameter-sharding, sequence-parallel and distributed-offload decision reductions remain in the hook. Peers take the maximum of skip (0), compute (1) and bypass (2); bypass runs uncached and clears local histories. Residual shape, device and dtype are checked before this agreement, so a peer cannot fall back to block execution after a shared skip decision. Offload groups are discovered through the model's declared block attributes using the shared offloader helper.

Calls with autograd enabled, missing branch context, missing or invalid scheduler metadata, conditioning-only vision or invalid indicators vote to bypass caching. A forward also votes to bypass if its scheduler step was not registered with `begin_step`. If metadata is unavailable or invalid at `begin_step`, the hook warns once and resets the evaluation history. The subsequent forwards still join the bypass vote and run uncached. All CFG ranks register the same global branch tuple, including idle ranks. Changes in active branches, nonconsecutive or repeated solver evaluations, and branch exceptions reset histories. Residuals remain separate for each named branch; no vote is taken across CFG branches.

## Scope and validation

This implementation adapts [#7939](https://github.com/vllm-project/vllm-omni/pull/7939) at `b9cadf2b47c40170558e59083f9dbcaaa9bdc4d7` to the forward protocol. It keeps the typed inputs and residual snapshot introduced by the migration. The target-only indicator and coordinated decisions intentionally change the old Transfer cache behavior; the PR's GPU quality results have not been reproduced on this protocol branch.

The image implementation proposed in [#6975](https://github.com/vllm-project/vllm-omni/pull/6975) uses a two-dimensional token grid, a leading grid-token count and model-derived sigma. Those fields are not added without an image-model consumer; extending SeaCache to those models requires reconciling their indicator representation and metadata source.

CPU tests cover branch separation, reset, fallbacks, protocol rejection, inherited methods, residual mutation, offload group discovery and the Cosmos3 execution boundary. GPU output quality, input-snapshot cost and distributed GPU compatibility require separate measurements; CPU checks do not establish those results.
