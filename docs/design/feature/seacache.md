# SeaCache forward protocol

SeaCache (Spectral-Evolution-Aware Cache) is a training-free caching method for diffusion models. It applies a filter designed to preserve content while suppressing noise, then compares the filtered inputs to decide when to reuse previous computations. See the [SeaCache repository](https://github.com/jiwoogit/SeaCache) for details.

In Cosmos3, the filter operates on the target latent, the model's internal representation of the image or video being generated. It uses the scheduler's noise level for the current step. For calls eligible for caching, the hook compares the accumulated change between filtered latents with a threshold. The first and last steps always compute, as do calls without residual history or after the configured limit on consecutive skips.

On a skipped step, the Cosmos3 hook estimates a residual from previously computed steps and adds it to the current block input. A residual is the difference between the block output and its input. The default uses linear extrapolation when enough history is available; otherwise it reuses the latest residual. SeaCache does not require fitted polynomial coefficients for the cache decision.

The reference implementations for Wan2.1, HunyuanVideo and FLUX filter the first block's modulated input and reuse a previous residual. Cosmos3 instead filters the target latent and supports residual extrapolation. The reference repository's quality and overhead measurements do not establish those results for Cosmos3.

Caching can change the generated output, so compare it with uncached generation for your workload. SeaCache is implemented for the Cosmos3 pipelines, including Edge through its inherited transformer methods. GPU measurements cover Nano only. Enable it with `cache_backend="sea_cache"`.

## Pipeline setup

The pipeline tells the hook which denoising step is running and which classifier-free guidance (CFG) branches will run. A branch is a separate model call, such as the conditional or unconditional call used for CFG.

Before running the branches:

1. Update the values returned by the hook's callbacks for the current step index, scheduler noise level (`sigma`) and total step count.
2. Call `hook.begin_step(branches)` with a tuple containing the active branch names. Do this before each group of CFG branch forwards, including repeated evaluations within the same denoising step.
3. Run each branch inside `hook.cache_context(name)` so the hook uses that branch's cached residuals.

When CFG runs across multiple ranks, every rank registers the same tuple of active branches. A rank must register the tuple even if it has no branch to run in that step.

The hook checks registration by step index. It cannot detect an omitted `begin_step` call when the same index is reused, so the pipeline must call it before every group of branch forwards.

## Model contract

A transformer implements `SupportsSeaCache` from `vllm_omni.diffusion.cache.seacache.protocol`. This interface includes the methods from `SupportsDecomposedForward` and adds `get_seacache_inputs(ctx) -> SeaCacheInputs`. It does not require TeaCache coefficients or defaults.

Cached and uncached execution use the same model methods:

1. `preprocess(..., skip_modulated_input=True)` prepares the block input and the other values the model needs, such as masks and positions.
2. `run_transformer_blocks(ctx)` runs the blocks and returns the updated state. It may return a new state object or change the existing hidden-state tensor in place.
3. `postprocess(ctx)` returns the model output. Everything it needs must be available after `preprocess`, because the blocks do not run on a cache hit.

`get_seacache_inputs` returns the unfiltered target in `latent`, with shape `(batch, channels, frames, height, width)`. It can also return a mask identifying which frames contain noise.

The hook filters the target latent to produce the **indicator**, the tensor it compares between steps. Cosmos3 includes any clean prefix frames in the target but excludes separate control hints from the indicator. The block input may also contain controls, action tokens and sound tokens, so it can differ from the indicator.

Setting `modulated_input=None` only disables TeaCache for that call. It does not disable SeaCache.

Hook installation rejects a model that does not implement the required methods. It also rejects a subclass that overrides `forward` while inheriting `preprocess` from a parent class, because the hook would skip the subclass's changes. SeaCache does not use extractor callbacks or read `CacheContext.extra_states`.

## Storing and reusing residuals

On a compute step, the hook copies the block input before running the blocks. It then stores `block_output - input_copy` before calling `postprocess`. Copying the input preserves its original value even if a block changes it in place.

For Cosmos3, the cached block output includes the final GEN normalization. On a cache hit, the hook estimates the residual from its saved history, adds it to the current block input and calls `postprocess`.

The hook keeps separate residuals for each named CFG branch. It clears its local branch histories when the active branch names change, steps repeat or run out of sequence, or a branch raises an exception. Repeating an evaluation within one denoising step therefore starts a new history.

## Keeping distributed ranks in agreement

Ranks that share block computation must agree on whether to run the blocks. Otherwise, one rank could wait for communication from another rank that skipped them. The hook coordinates decisions across parameter-sharding, sequence-parallel and distributed-offload groups.

Each rank chooses one of the following decisions. The group takes the highest value:

| Decision | Value | Action |
| --- | ---: | --- |
| Skip | 0 | Reuse an estimated residual. |
| Compute | 1 | Run the blocks and record a new residual. |
| Bypass | 2 | Clear local histories and run without caching. |

Within each reduction group, a compute or bypass decision takes precedence over a skip. Before the reduction, the hook checks cached residual shape, device and dtype. An incompatible residual clears that branch's history and forces computation.

The hook uses the model's declared block attributes and the shared offloader helper to find distributed-offload groups. It does not combine decisions across the CFG group itself.

## Calls that run without caching

A rank chooses bypass when:

- Gradient tracking is enabled.
- No CFG branch name has been set with `cache_context`.
- The scheduler information is missing or invalid.
- The supplied frame mask marks no noisy vision frames.
- The target latent cannot be filtered, for example because it has an unsupported shape.
- The current step was not registered with `begin_step`.

If scheduler information is missing or invalid at `begin_step`, the hook logs a warning once per distinct message and clears the saved history. Later forwards still take part in the group decision, with caching disabled until a valid step is registered.

An incompatible indicator history or a nonfinite comparison distance forces computation rather than bypass. Errors from model preprocessing, block execution or postprocessing still propagate to the caller.

## Scope and validation

Measurements on one MI300X cover Cosmos3 Nano text-to-video and Transfer with one or two controls, using sequential CFG. At 1280×720, 93 frames and 35 steps, text-to-video requests completed 1.81× faster with the default SeaCache settings. That ratio uses total request time, averaged over two repeats for each of two prompts. Cached outputs showed visible changes in position and motion. Super, Edge, image generation and distributed GPU execution remain unmeasured. The cost of copying block inputs has not been measured separately.

The current interface expects a five-dimensional target latent, including for Cosmos3 image generation. Models that store image latents as a sequence of tokens need an adapter or a different input interface; the hook cannot filter that layout directly.

CPU tests check that branches keep separate state, histories reset when needed and ineligible calls run uncached. They also cover model interface checks, inherited methods, blocks that modify inputs in place, discovery of offload groups and the cached block range in Cosmos3. CPU tests do not establish output quality or GPU performance.
