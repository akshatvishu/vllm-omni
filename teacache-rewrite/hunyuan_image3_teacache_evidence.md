# Hunyuan Image 3 TeaCache port: evidence and design

This document records the evidence for the Hunyuan Image 3 TeaCache port. It separates facts from design decisions. A design decision is marked as an inference when the code and the paper support the decision but do not state it directly.

No Hunyuan Image 3 benchmark was run while preparing this document. The document therefore makes no claim about Hunyuan speedup, image quality, or memory use.

## Sources

The repository worktree used for the code references is `teacache_rewrite_base` at `a4352eae`.

The refactor branch used for the TeaCache contract references is `origin/teacache_alex_rewrite` at `263ae466`.

The paper source is [teacache.pdf](/home/aja/vllm-omni/teacache.pdf). The paper is `Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model`, arXiv:2411.19108v2.

The relevant code files are:

* [Hunyuan transformer](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2024)
* [Hunyuan pipeline](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:336)
* [TeaCache refactor proposal](/home/aja/vllm-omni/teacache-rewrite/teacache_refactor_proposal.md:140)

For files that exist only on the refactor branch, the source reference uses this form:

```text
origin/teacache_alex_rewrite@263ae466:<repository path>:<line>
```

The exact source can be inspected with `git show`.

## The recommended implementation boundary

The recommended design is to attach the new TeaCache runtime to `HunyuanImage3Model` and call it around the model's `self.layers` loop. The pipeline should continue to prepare embeddings and KV state, call the model, run `ragged_final_layer`, apply CFG, and step the scheduler.

The code proves that `HunyuanImage3Pipeline` owns the transformer instance. The pipeline assigns `self.model = HunyuanImage3Model(...)` and `self.transformer = self.model` at [pipeline_hunyuan_image3.py:386](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:386).

The code proves that `HunyuanImage3Model.forward` owns the decoder layer loop at [hunyuan_image3_transformer.py:2485](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2485).

The code proves that `HunyuanImage3Pipeline.forward_call` calls the model at [pipeline_hunyuan_image3.py:1729](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1729), then runs `ragged_final_layer` at [pipeline_hunyuan_image3.py:1767](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1767).

The paper supports caching a residual inside a diffusion transformer. Section 3.4 on PDF page 6 defines the cached residual as output minus input and says that the next model output is updated from the current input. The paper does not specify Hunyuan's code boundary, so the exact boundary is a repository design decision.

The resulting call structure is:

```text
HunyuanImage3Pipeline.forward_call
  prepare image and timestep embeddings
  prepare ImageKVCacheManager state
  HunyuanImage3Model.forward
    sequence parallel preprocessing
    TeaCacheBlockExecutor around self.layers
    sequence parallel postprocessing
  ragged_final_layer
  CFG
  scheduler
```

The final layer remains outside the cached region because the pipeline calls it after the model returns. Recomputing it is an inference from the existing call order and from the paper's residual cache definition. It is not a measured quality result.

## The refactor contract replaces the old protocol

The refactor branch defines `TeaCacheBlockExecutor.run` with four inputs: `modulated_input`, `residual_inputs`, `compute_fn`, and `do_true_cfg`. The definition is at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/interface.py:13`.

The refactor branch validates `supports_teacache`, `tea_cache_model_key`, `tea_cache_executor`, and `get_teacache_coefficients`. The validation is at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/interface.py:39`.

The refactor branch test describes the intended model contract as validation without Protocol inheritance at `origin/teacache_alex_rewrite@263ae466:tests/diffusion/cache/test_teacache_protocol.py:64`.

The current worktree still has the old Hunyuan stubs. They are `preprocess`, `run_transformer_blocks`, and `postprocess` at [hunyuan_image3_transformer.py:2368](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2368). The target refactor branch's Hunyuan model starts at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2018` and does not contain those old methods.

The implementation should therefore add the new model attributes and call the new executor. It should not add a Hunyuan `ForwardState` dataclass or implement the old three method protocol.

## The runtime caches a block residual, not the final diffusion prediction

The refactor runtime clones the declared residual inputs, calls `compute_fn`, checks output arity and shape, and stores `output - input`. The code is at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:105`.

On a cache hit, the runtime returns the current boundary input plus the stored residual. The code is at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:124`.

The paper gives the same residual definition in Section 3.4 on PDF page 6.

The existing Hunyuan cache has a different behavior. It stores the final `diffusion_prediction` at [hunyuan_image3_transformer.py:3172](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3172) and returns that prediction on a hit at [hunyuan_image3_transformer.py:3174](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3174).

The native port must therefore pass the hidden state before the decoder layers as `residual_inputs`, and the `compute_fn` must return the hidden state after the decoder layers. Returning `diffusion_prediction` from that executor would violate the refactor runtime's boundary contract. This conclusion follows from the runtime shape and residual checks and from the existing Hunyuan call order.

## Sequence parallelism compatibility

The native boundary is compatible with Hunyuan sequence parallelism in principle. The current code does not prove the complete TeaCache and sequence parallel combination, so it requires a distributed test before it is enabled as a supported combination.

Hunyuan's `_sp_plan` splits the image related outputs of `pre_processor` along sequence dimension 1 and gathers the output of `post_processor` along sequence dimension 1. The plan is at [hunyuan_image3_transformer.py:2032](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2032).

The sequence parallel type definitions say that `SequenceParallelInput` splits a tensor along the configured dimension and that `SequenceParallelOutput` gathers a tensor across ranks. The definitions are at [sp_plan.py:206](/home/aja/vllm-omni/vllm_omni/diffusion/distributed/sp_plan.py:206) and [sp_plan.py:253](/home/aja/vllm-omni/vllm_omni/diffusion/distributed/sp_plan.py:253).

Hunyuan calls `pre_processor` only when the sequence parallel world size is greater than one at [hunyuan_image3_transformer.py:2421](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2421). It then runs the decoder layer loop at [hunyuan_image3_transformer.py:2485](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2485), and calls `post_processor` after the loop at [hunyuan_image3_transformer.py:2521](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2521).

The executor should therefore be placed after Hunyuan's sequence parallel preprocessing and before its sequence parallel postprocessing:

```python
# Existing preprocessing and padding happen before this point.

if cacheable:
    (hidden_states,) = self.tea_cache_executor.run(
        modulated_input=tea_cache_modulated_input,
        residual_inputs=(hidden_states,),
        compute_fn=run_transformer_blocks,
        do_true_cfg=tea_cache_do_true_cfg,
    )
else:
    (hidden_states,) = run_transformer_blocks()

# Existing text/image split and post_processor gather happen after this point.
```

The refactor runtime stores the residual tensors on the process that calls it. It checks that every computed output has the same shape as the local residual input at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:105`. The runtime computes the cache decision locally at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:68` and does not perform a sequence parallel collective.

The code supports a per rank residual cache because each rank receives a local sequence shard before the decoder layers and the gathered result is needed only after the decoder layers. The paper supports this residual form in Section 3.4 on PDF page 6. The per rank conclusion is an implementation inference, not a result reported by the paper.

The cache decision must use the same timestep embedding on every sequence parallel rank. Passing a local hidden state as `modulated_input` could produce different decisions because the runtime calculates the metric locally. The existing Hunyuan cache already uses the timestep embedding at [hunyuan_image3_transformer.py:3159](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3159), and the paper identifies timestep embedding as independent of the noisy input and text embedding in Section 3.2 on PDF page 4.

The following guards are required for sequence parallelism:

* Skip TeaCache on the first image step. Hunyuan uses a prompt partition on the first step and `prompt_len = 0` later at [hunyuan_image3_transformer.py:1974](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1974).
* Keep `uncond_cfg_prefill` outside TeaCache. The separate prefill path is at [hunyuan_image3_transformer.py:1134](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1134).
* Keep the existing `query_lens` and `seq_lens` equality assertion. Hunyuan requires those values to match across the sequence parallel batch at [hunyuan_image3_transformer.py:2446](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2446).
* Keep the existing auto padding and attention mask handling. Hunyuan computes shard padding and extends the attention mask at [hunyuan_image3_transformer.py:2452](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2452).

The paper does not discuss sequence parallelism or rank synchronization. A successful single rank test does not prove that sequence parallelism is safe. The required proof is a multi rank test that checks identical cache decisions, no collective hang, output shape after `post_processor`, and image output parity against the no cache sequence parallel path.

## Qwen Image is the closest repository example

Qwen Image uses the same structural placement that Hunyuan should use. The Qwen implementation is evidence for the placement, but it is not evidence that Hunyuan sequence parallelism has already been validated.

The Qwen transformer declares the native TeaCache attributes at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py:935`.

The Qwen `_sp_plan` shards `image_rope_prepare` outputs and gathers `proj_out` output at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py:958`.

Qwen runs `image_rope_prepare` before the transformer block closure at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py:1130` and defines the transformer block closure at line 1201.

Qwen calls the TeaCache executor after the SP preparation and before `norm_out` and `proj_out`. It passes both image and text hidden states as residual inputs at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py:1215`.

Qwen's `proj_out` remains after the executor, and the code states that its SP gather is handled by the `_sp_plan` hook at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py:1229`.

The corresponding Hunyuan placement is therefore:

```text
Hunyuan pre_processor
TeaCache around Hunyuan self.layers
Hunyuan post_processor
```

Qwen passes `do_true_cfg=self.do_true_cfg` into the executor at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py:1222`. Hunyuan should pass its explicit CFG mode in the same way.

Qwen does not prove that a local hidden state is safe as a distributed cache metric. Qwen computes `modulated_input` from the local `hidden_states` at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py:1218`, while the refactor runtime computes the decision locally and has no collective at `runtime.py:68`. The target Qwen TeaCache test uses one GPU at `origin/teacache_alex_rewrite@263ae466:tests/diffusion/cache/test_teacache_unit.py:198`, and the repository has no Qwen TeaCache sequence parallel test in the target branch.

The Qwen forward path has image sequence inputs and no Hunyuan style `first_step` prompt transition. Its input preparation describes `hidden_states` as an image sequence at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py:97`, and its forward signature at line 1084 has no `first_step` argument. Hunyuan must retain its first step guard because its preprocessor changes `prompt_len` between first and later steps.

## The timestep embedding is the supported cache metric

The current Hunyuan cache uses a timestep embedding as its cache decision input. The code computes `self.model.time_embed(t.unsqueeze(0))` at [hunyuan_image3_transformer.py:3159](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3159).

The Hunyuan pipeline owns the `time_embed` module at [pipeline_hunyuan_image3.py:413](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:413). On later image steps, the pipeline already computes `t_emb = self.time_embed(timestep)` and uses it for `patch_embed` at [pipeline_hunyuan_image3.py:1705](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1705).

The paper says in Section 3.2 on PDF page 4 that text embedding stays constant across denoising and is excluded from the difference metric. The paper then selects timestep embedding modulated noisy input as the input indicator. Section 3.3 on PDF page 5 describes polynomial rescaling of the input difference.

The clean implementation is to pass the already computed timestep embedding from the pipeline to the model as a model specific `tea_cache_modulated_input` argument. The pipeline should not compute a second embedding. The coefficient fit must use the same metric that the runtime receives.

The paper does not prove that the Hunyuan timestep embedding is sufficient for Hunyuan Image 3. The old Hunyuan implementation provides a codebase precedent, and the new coefficients still require measurement.

## The first step must be outside the cached region

The Hunyuan pipeline builds the first image input by inserting image and timestep tokens into the prompt embeddings at [pipeline_hunyuan_image3.py:1699](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1699). On later steps it builds a new timestep embedding and image embedding and concatenates them at [pipeline_hunyuan_image3.py:1705](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1705).

The sequence parallel preprocessor makes the difference explicit. It sets a prompt length from the generated image position on the first step and sets `prompt_len = 0` on later steps at [hunyuan_image3_transformer.py:1974](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1974).

The refactor runtime checks that a newly computed output has the same shape as its residual input. The check is at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:117`. The current target runtime decides whether to compute at lines 102 to 105 and applies a cached residual at lines 124 to 127, so it does not check the current residual shape before a cache hit.

The safe first implementation should not call the TeaCache executor when `first_step` is true. The first later step becomes the first TeaCache miss and seeds the residual with the stable image step shape. This is a code based inference. The TeaCache paper does not discuss Hunyuan's prompt and image token transition.

The Hunyuan final layer also uses different output selection on the first step and later steps. It selects `image_mask` on the first step and `x[:, 1:, :]` later at [pipeline_hunyuan_image3.py:1037](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1037). This is additional evidence that the first step must remain an explicit path.

## Image KV state must remain outside TeaCache

Hunyuan has a dedicated `ImageKVCacheManager` at [hunyuan_image3_transformer.py:930](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:930).

The manager clears its prompt cache and caches prompt KV on the first image step at [hunyuan_image3_transformer.py:1143](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1143).

The manager reuses prompt KV on later image steps at [hunyuan_image3_transformer.py:1162](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1162).

The manager has a separate negative CFG prefill path at [hunyuan_image3_transformer.py:1134](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1134).

The TeaCache paper does not describe this Hunyuan KV manager. The code proves that prompt KV setup and negative prefill have separate state transitions. The native implementation must keep those transitions in the existing attention path and must not use TeaCache for `uncond_cfg_prefill`.

## Cache only image denoising

The outer forward path has separate branches for `gen_text`, `gen_image`, and `uncond_cfg_prefill` at [pipeline_hunyuan_image3.py:1692](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1692) and [pipeline_hunyuan_image3.py:1752](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1752).

The outer forward path produces no diffusion prediction for text generation or negative CFG prefill at [pipeline_hunyuan_image3.py:1752](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1752).

The executor should therefore be enabled only when all of the following conditions hold:

```python
mode == "gen_image"
and not first_step
and not uncond_cfg_prefill
and not output_attentions
and not output_hidden_states
```

The first three conditions follow directly from the Hunyuan control flow. The last two conditions are design guards because the TeaCache runtime returns only the declared block outputs, while the model's optional attention and hidden state collections are produced inside the skipped layer loop at [hunyuan_image3_transformer.py:2485](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2485).

## CFG behavior must use the refactor flag

The current pipeline uses sequential CFG by duplicating the latent batch at [hunyuan_image3_transformer.py:3151](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3151).

The current pipeline uses CFG parallel by sending one branch to each rank at [hunyuan_image3_transformer.py:3147](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3147).

The refactor runtime uses `do_true_cfg` to select branch state. With `do_true_cfg=True`, it selects positive or negative state from the CFG rank when the CFG world size is greater than one. Otherwise it alternates positive and negative state by forward count. The code is at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:54`.

The correct mapping follows from these two code paths:

| Hunyuan execution path | `do_true_cfg` | Evidence |
| --- | --- | --- |
| No CFG | `False` | One model call has one branch. The runtime's default branch is positive at `runtime.py:66`. |
| Sequential CFG | `False` | Both branches are one model input batch, so the runtime must cache the complete batch as one boundary. The batch construction is at `hunyuan_image3_transformer.py:3151`. |
| CFG parallel | `True` | Each rank owns one branch, and the runtime has rank based branch selection at `runtime.py:62`. |

The paper does not define this vLLM CFG contract. The mapping is a direct consequence of the refactor runtime and Hunyuan's current batch layout.

## The cache metric must be the same on every rank

The refactor runtime computes the relative distance from the local `modulated_input` and does not perform a collective decision. The local calculation is at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:68`.

Hunyuan has sequence parallel preprocessing and postprocessing declared in `_sp_plan` at [hunyuan_image3_transformer.py:2032](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2032).

The safest metric is the timestep embedding computed from the denoising timestep, because the old Hunyuan cache also uses it at [hunyuan_image3_transformer.py:3159](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3159). This avoids using a rank local hidden state for the decision. The rank invariant conclusion is an implementation inference from the local runtime calculation and the timestep only input.

The paper says in Section 3.2 on PDF page 4 that timestep embedding is independent of the noisy input and text embedding. The paper does not discuss distributed rank synchronization.

## Backend wiring must use the generic native path

The Hunyuan pipeline already exposes the transformer under `pipeline.transformer` at [pipeline_hunyuan_image3.py:386](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:386).

The generic refactor backend validates the transformer, resolves coefficients from the model, creates `TeaCacheRuntime`, attaches it to `transformer.tea_cache_executor`, and records the runtime for refresh. The code is at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/backend.py:111`.

The target branch still has a Hunyuan custom enabler that stores `_tea_cache_config` on the pipeline instead of installing a runtime. The code is at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/backend.py:44`.

The target branch also skips runtime reset for that Hunyuan custom configuration at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/backend.py:137`.

The native Hunyuan port should add the new capability attributes to `HunyuanImage3Model`, remove Hunyuan from `CUSTOM_TEACACHE_ENABLERS`, and let the generic backend install and refresh the runtime. This is a direct application of the refactor backend contract. The old custom enabler must not remain active together with the new pipeline cache loop.

## Step execution should remain disabled in the first port

The current Hunyuan pipeline rejects every nonempty diffusion cache backend during step execution at [pipeline_hunyuan_image3.py:503](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:503).

The step implementation can group multiple request states. It validates a group at [pipeline_hunyuan_image3.py:2009](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:2009) and runs the merged group at [pipeline_hunyuan_image3.py:2097](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:2097).

The refactor runtime stores one `TeaCacheRuntimeState` containing positive and negative branch state. The state has no request identifier. The code is at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:35`.

The current code therefore has model scoped TeaCache state and request grouped step execution. Supporting both requires request scoped runtimes or a request key in the runtime state. The paper has no serving lifecycle model. Keeping the existing rejection is the smallest change supported by the current code.

## `use_cache` needs an explicit decision

The Hunyuan model falls back to `self.config.use_cache` when `use_cache` is not passed at [hunyuan_image3_transformer.py:2403](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2403).

When `use_cache` is true, the layer loop records `next_decoder_cache` at [hunyuan_image3_transformer.py:2512](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2512) and returns it at [hunyuan_image3_transformer.py:2529](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2529).

The refactor executor contract returns only the tensors declared in `residual_inputs`. It has no field for `past_key_values` at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/interface.py:16`.

The step preparation path already sets `model_kwargs["use_cache"] = False` at [pipeline_hunyuan_image3.py:1884](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1884).

The native implementation must either use TeaCache only when image denoising has `use_cache=False`, or add a separate mechanism that preserves the required `past_key_values` on a cache hit. The paper does not address Hugging Face cache metadata. The first option is the smaller change, but baseline behavior must be checked before changing normal non step execution.

## Coefficients must be fit for the new boundary

The target refactor configuration requires exactly five finite polynomial coefficients. The validation is at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/config.py:22`.

The target coefficient estimator collects the modulated input and model output at the native boundary at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/coefficient_estimator.py:21`.

The estimator calculates relative L1 differences and fits a fourth order polynomial with `np.polyfit` at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/coefficient_estimator.py:140` and `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/coefficient_estimator.py:147`.

The paper defines the relative L1 metric in Eq. 4 on PDF page 4 and polynomial rescaling in Eq. 6 and Eq. 7 on PDF page 5.

The current Hunyuan coefficients are attached to the old pipeline prediction cache at [hunyuan_image3_transformer.py:2378](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2378) and used with the old final prediction path at [hunyuan_image3_transformer.py:3172](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3172).

The code and paper provide no evidence that those coefficients remain valid after changing the cached output from final prediction to decoder layer residual. The coefficients should therefore be collected again after the native boundary is implemented.

## Tests required by the evidence

The refactor unit tests cover runtime behaviors that Hunyuan will use. The target branch lists protocol validation, residual decision math, shape and arity checks, CFG state isolation, reset, and invalid numeric input at `origin/teacache_alex_rewrite@263ae466:tests/diffusion/cache/test_teacache_unit.py:7`.

The target branch tests first step computation at `origin/teacache_alex_rewrite@263ae466:tests/diffusion/cache/test_teacache_unit.py:161`, cache hit and miss behavior at line 183, and shape mismatch rejection at line 268.

The target branch native model test proves that a repeated model call can avoid calling the transformer blocks while returning the same output for its controlled test input at `origin/teacache_alex_rewrite@263ae466:tests/diffusion/cache/test_teacache_unit.py:203`.

The repository Hunyuan tests are under `tests/diffusion/models/hunyuan_image3/`, including [step execution](/home/aja/vllm-omni/tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py:1). The offline test is [test_hunyuanimage3.py](/home/aja/vllm-omni/tests/e2e/offline_inference/test_hunyuanimage3.py:1), and the accuracy test is [test_hunyuan_image3.py](/home/aja/vllm-omni/tests/e2e/accuracy/test_hunyuan_image3.py:1).

The Hunyuan native port should add tests for the following code paths:

* `gen_text` must not update TeaCache state.
* `uncond_cfg_prefill` must not update TeaCache state.
* The first image step must run the decoder layers.
* The first later image step must seed the residual with the stable shape.
* A repeated timestep must skip the decoder layers when the threshold allows it.
* A threshold crossing must run the decoder layers and replace the residual.
* Sequential CFG must use one combined branch state.
* CFG parallel must use the rank branch state.
* Runtime refresh must clear state between requests.
* Shape changes must not reuse a prior residual.
* The `use_cache` behavior must be explicit.

The first seven items are supported by the Hunyuan control flow and the refactor runtime checks. The remaining items require additional lifecycle, distributed, and cache state validation in Hunyuan. The paper does not provide Hunyuan test cases.

## Test plan for the migration

The recommended test order is CPU contract tests, a CPU Hunyuan boundary test, a one GPU tiny Hunyuan model test, separate two GPU SP and CFG tests, and finally a real weight image test. Each stage proves a different part of the port. A passing TeaCache runtime test does not prove that Hunyuan skips its decoder layers, and a passing one GPU test does not prove that sequence parallel ranks make the same cache decision.

### Keep the existing CPU tests as the first gate

Run the target refactor unit file before adding Hunyuan specific assertions:

```text
.venv/bin/pytest --run-level core_model -q tests/diffusion/cache/test_teacache_unit.py
```

The target file is marked `core_model` and `cpu` at `origin/teacache_alex_rewrite@263ae466:tests/diffusion/cache/test_teacache_unit.py:34`. It already tests the new protocol and configuration at lines 77 to 153, first compute and hit or miss behavior at lines 161 to 224, input snapshot behavior at lines 227 to 248, arity and shape validation at lines 251 to 282, numeric guards at lines 285 to 332, coefficient collection at lines 335 to 349, and CFG state and reset at lines 357 to 451.

Add one generic regression to this file before relying on it for Hunyuan. Call the runtime once with a residual of shape `N`, then call it with the same modulated input and a residual of shape `M`. The expected behavior should be a forced compute with the new shape, or a clear error before the cache hit. The current target runtime can otherwise reach the cache hit at lines 124 to 127 before it checks output shape at lines 117 to 119. Hunyuan needs this guard because its first step has a prompt shape and later steps have an image only shape, as shown by the pipeline at [pipeline_hunyuan_image3.py:1699](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1699) and [pipeline_hunyuan_image3.py:1705](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1705).

Do not use the current [test_teacache.py](/home/aja/vllm-omni/tests/diffusion/cache/test_teacache.py:1) as the migration contract. That file is a hardware smoke test for the old system path and calls `OmniRunner` with a Qwen random weight model at lines 20 to 39. The refactor unit file and its `TeaCacheBlockExecutor` calls are the relevant contract.

### Preserve the Hunyuan CPU coverage that already exists

Run the existing Hunyuan CPU component tests without changing their purpose:

```text
.venv/bin/pytest --run-level core_model -q \
  tests/diffusion/models/hunyuan_image3/test_image_kv_cache_manager.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py
```

The ImageKVCacheManager file is marked `core_model` and `cpu` at [test_image_kv_cache_manager.py:18](/home/aja/vllm-omni/tests/diffusion/models/hunyuan_image3/test_image_kv_cache_manager.py:18). It already covers basic prompt cache and reuse for batch sizes one and two at lines 124 to 188, AR KV with sequence parallel size one and two at lines 195 to 307, sequential and parallel CFG setup at lines 314 to 450, and cross request isolation at lines 457 to 500. Those tests prove the KV state transitions that the TeaCache port must leave intact.

The step execution file is also marked `core_model`, `diffusion`, and `cpu` at [test_hunyuan_image3_step_execution.py:25](/home/aja/vllm-omni/tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py:25). Keep its rejection of cache backends aligned with [pipeline_hunyuan_image3.py:489](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:489), because the step path groups request states while the target runtime owns one model state object at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:35`.

The existing KV tests do not call `HunyuanImage3Model.forward`, and the existing step tests replace `forward_call` with a fake function at [test_hunyuan_image3_step_execution.py:351](/home/aja/vllm-omni/tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py:351). Add the native TeaCache model tests separately so the KV and step tests remain focused on their current contracts.

### Add a Hunyuan protocol and backend test

Extend the target protocol file after the Hunyuan model has the new native attributes:

```text
.venv/bin/pytest --run-level core_model -q tests/diffusion/cache/test_teacache_protocol.py
```

The target protocol file checks that native models expose the cache boundary and five coefficients at `origin/teacache_alex_rewrite@263ae466:tests/diffusion/cache/test_teacache_protocol.py:37` and lines 64 to 68. Add `HunyuanImage3Model` to that native model list. Replace the old Hunyuan custom configuration assertion at lines 126 to 142 with a generic native backend assertion.

The Hunyuan backend test should create a small fake pipeline with `transformer` set to a Hunyuan boundary object. It should call `TeaCacheBackend.enable`, assert that `transformer.tea_cache_executor` is a `TeaCacheRuntime`, assert that one runtime is recorded, set `forward_cnt` to a nonzero value, call `refresh`, and assert that the counter returns to zero. The target generic backend installs the runtime at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/backend.py:111` to `128` and resets installed runtimes at lines 137 to 148. The old Hunyuan custom enabler only stores `_tea_cache_config` at lines 44 to 51, so this test catches an incomplete migration.

The runner calls cache refresh before a normal request batch at [diffusion_model_runner.py:373](/home/aja/vllm-omni/vllm_omni/diffusion/worker/diffusion_model_runner.py:373) and [diffusion_model_runner.py:461](/home/aja/vllm-omni/vllm_omni/diffusion/worker/diffusion_model_runner.py:461). The backend test and one request lifecycle test must therefore prove that a second request cannot reuse the first request's residual.

### Add a CPU Hunyuan boundary test

Add `tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_teacache.py` with `core_model`, `diffusion`, and `cpu` marks for the structural tests. Use the real `HunyuanImage3Model.forward` and a small `HunyuanImage3Config`, but replace the decoder layers with deterministic counting layers before the forward call. The model constructor creates `self.layers` through `make_layers` at [hunyuan_image3_transformer.py:2043](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2043), and the forward method owns the decoder loop at [hunyuan_image3_transformer.py:2485](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2485). The configuration exposes the small model fields needed for this fixture at [hunyuan_image3_transformer.py:1295](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1295).

Patch the layer factory before constructing the model if CPU vLLM attention construction requires an unavailable accelerator. Keep the real `HunyuanImage3Model.forward` in the test. Replacing the entire model forward would only test a copy of the proposed code and would not prove the placement around `self.layers`.

Use a deterministic layer that returns one hidden state with the same shape as its input and increments a counter. Attach a `TeaCacheRuntime` with a polynomial that makes equal modulated inputs hit. Test the following cases:

* The first later image step calls every counting layer and seeds one residual. The second later image step with the same shape and modulated input calls no layer and returns the current hidden state plus the stored residual. The runtime stores `output - input` and applies the residual on a hit at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:105` to `133`, and the paper defines the same residual in Section 3.4 on PDF page 6.
* A changed modulated input crosses the threshold and calls every layer again. Assert that the new residual replaces the old residual by making the counting layer return a different known delta.
* `gen_text`, `uncond_cfg_prefill`, and `first_step=True` call the model layers but do not increment TeaCache state. The pipeline has separate branches for those modes at [pipeline_hunyuan_image3.py:1692](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1692), [pipeline_hunyuan_image3.py:1699](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1699), and [pipeline_hunyuan_image3.py:1757](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1757).
* A first image step followed by a later image step seeds the runtime only on the later step. Assert the first step's prompt shaped hidden state never becomes the cached residual for the image only shape. Hunyuan changes `prompt_len` from a first step value to zero later at [hunyuan_image3_transformer.py:1974](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1974).
* A boundary shape change after a cached later step follows the explicit shape policy. The recommended policy is to force a compute and replace the residual. The generic runtime test described above must enforce this before the Hunyuan model test runs.
* `use_cache=True` follows the chosen explicit policy. The first port should bypass TeaCache or raise a clear error when a past key value would be lost, because the Hunyuan model records and returns decoder cache state at [hunyuan_image3_transformer.py:2512](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2512) and [hunyuan_image3_transformer.py:2529](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2529), while the refactor executor returns only declared residual tensors at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/interface.py:16`.

Add one pipeline handoff test to the same file. Replace `time_embed`, `patch_embed`, and the model with recording fakes, run a later image step, and assert that the timestep embedding passed to `patch_embed` is the same tensor passed as the TeaCache metric to the model. The pipeline already computes `t_emb` before `patch_embed` at [pipeline_hunyuan_image3.py:1705](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:1705), and the old Hunyuan cache uses the model timestep embedding at [hunyuan_image3_transformer.py:3159](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3159). The test prevents a second, different metric from being introduced during the port.

### Add a one GPU tiny Hunyuan model test

The target refactor's tiny model test is the right model test pattern, but it currently covers Qwen only. It builds a two layer model, seeds finite parameters, attaches `TeaCacheRuntime`, hooks each transformer block, makes two identical forward calls, and asserts that the second call skips the blocks at `origin/teacache_alex_rewrite@263ae466:tests/diffusion/cache/test_teacache_protocol.py:169` to `231`. The test is explicitly hardware gated to one CUDA or ROCm card at lines 198 to 203, so it is not a CPU proof.

Add the same test for `HunyuanImage3Model` in the Hunyuan test file. Construct a two layer `HunyuanImage3Config` with one expert and small hidden dimensions, initialize it under `set_current_vllm_config(VllmConfig())`, seed every parameter with finite values, and use `set_forward_context` during the call. Use `mode="gen_image"`, `first_step=False`, `use_cache=False`, equal `query_lens` and `seq_lens`, and a small stable input shape. Hook `model.layers` and assert the following:

* The first call invokes every layer.
* The second call with the same hidden input and timestep metric does not increase the layer counter.
* `state.forward_cnt` is two and both outputs are equal.
* A changed timestep metric causes the layer counter to increase.
* A first step and an `uncond_cfg_prefill` call do not touch the executor.

The assertions mirror the target Qwen test at lines 203 to 228, but the Hunyuan call must also pass its `first_step`, `mode`, KV lengths, and image token arguments because those values are part of the Hunyuan model forward signature at [hunyuan_image3_transformer.py:2381](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2381). The tiny test proves the actual Hunyuan forward boundary. It does not prove image quality or the complete pipeline.

### Test the Hunyuan KV manager and TeaCache together

Use hooks on the tiny Hunyuan decoder attention or its `ImageKVCacheManager` to test the state interaction. The first image step must clear and cache prompt KV at [hunyuan_image3_transformer.py:1143](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1143). A later miss must reuse the prompt KV at [hunyuan_image3_transformer.py:1162](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1162). A TeaCache hit must skip the decoder layers and therefore must not call the attention manager. A later forced miss must call the manager again and must preserve the same prompt cache.

Run the sequence `first step`, `later miss`, `later hit`, `later miss`. Assert the manager call count, cached prompt length, hidden output shape, and decoder layer count after each call. Keep the existing direct manager tests as the lower level proof because their SP cases intentionally call `_cache_prompt_kv` and `_reuse_prompt_kv` directly to avoid full SP infrastructure at [test_image_kv_cache_manager.py:219](/home/aja/vllm-omni/tests/diffusion/models/hunyuan_image3/test_image_kv_cache_manager.py:219).

### Test sequence parallelism separately from single GPU behavior

The Hunyuan SP plan splits inputs in `pre_processor` and gathers outputs in `post_processor` at [hunyuan_image3_transformer.py:2032](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2032). The forward method performs preprocessing before the layer loop at [hunyuan_image3_transformer.py:2421](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2421), and performs postprocessing after the loop at [hunyuan_image3_transformer.py:2521](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2521). Put the executor between those points.

The current CPU KV test with `sp_size=2` proves manager shape and KV behavior, but its docstring says that it calls private manager methods directly to avoid full SP infrastructure at [test_image_kv_cache_manager.py:195](/home/aja/vllm-omni/tests/diffusion/models/hunyuan_image3/test_image_kv_cache_manager.py:195) and [test_image_kv_cache_manager.py:219](/home/aja/vllm-omni/tests/diffusion/models/hunyuan_image3/test_image_kv_cache_manager.py:219). The SP hook unit tests prove plan and hook registration for a small model at [test_sp_plan_hooks.py:850](/home/aja/vllm-omni/tests/diffusion/distributed/test_sp_plan_hooks.py:850) to [test_sp_plan_hooks.py:878](/home/aja/vllm-omni/tests/diffusion/distributed/test_sp_plan_hooks.py:878). Neither test proves Hunyuan TeaCache under distributed execution.

Add a two GPU hardware test using a tiny Hunyuan model and the real Hunyuan `_sp_plan`. Run the same later image sequence once without TeaCache and once with TeaCache. Test both an image token count that divides by two and a count that needs the existing auto padding. The existing SP correctness test uses two card hardware marks and compares an SP result with a baseline at [test_sequence_parallel.py:173](/home/aja/vllm-omni/tests/diffusion/distributed/test_sequence_parallel.py:173) to [test_sequence_parallel.py:239](/home/aja/vllm-omni/tests/diffusion/distributed/test_sequence_parallel.py:239).

The distributed test must assert all of the following:

* Both ranks complete the first step and the later miss without a collective hang.
* Both ranks observe the same sequence of compute and hit decisions. Gather the local compute counters and assert equal values.
* Both ranks receive the same timestep metric before the local runtime decision. The runtime calculates the metric locally and performs no collective at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:68` to `88`, so a rank local hidden state must not be used as the decision input.
* The later hit skips all local decoder layers on both ranks.
* The gathered output has the original sequence shape after `post_processor` and matches the no cache SP output within the selected numerical tolerance.

Do not combine SP and CFG parallel in the first distributed test. Add a separate two rank CFG parallel test with `do_true_cfg=True`, then combine the groups only after both tests pass. The target runtime selects a branch from CFG rank when the CFG world size is greater than one at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:54` to `65`. Hunyuan sends one branch per rank in its CFG parallel path at [hunyuan_image3_transformer.py:3147](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3147), while sequential CFG duplicates the batch at [hunyuan_image3_transformer.py:3151](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3151).

For CFG parallel, run one positive and one negative branch on two ranks, make the layer return different known branch deltas, and then repeat the same timestep. Assert that rank zero reuses only the positive residual and rank one reuses only the negative residual. Keep `do_true_cfg=False` for sequential CFG and assert that the complete batch is one boundary tensor. The target runtime unit tests already isolate the sequential branches at `origin/teacache_alex_rewrite@263ae466:tests/diffusion/cache/test_teacache_unit.py:357` to `426`; the Hunyuan test must prove the Hunyuan call wiring.

### Test lifecycle and unsupported paths

Add a request lifecycle test that runs one normal generation, calls the backend refresh, and runs a second generation with a different prompt or shape. Assert that the first call of the second generation executes the decoder layers. The runner refreshes the backend before calling `pipeline.forward` at [diffusion_model_runner.py:461](/home/aja/vllm-omni/vllm_omni/diffusion/worker/diffusion_model_runner.py:461), and the refactor backend resets each installed runtime at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/backend.py:147` to `148`.

Add negative tests for cache state in `gen_text`, `uncond_cfg_prefill`, and step execution. Hunyuan step execution rejects sequence parallel, CFG parallel, and diffusion cache backends at [pipeline_hunyuan_image3.py:498](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:498) to [pipeline_hunyuan_image3.py:505](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:505). The first TeaCache port should keep those rejections until request scoped cache state exists.

### Test coefficient collection at the new boundary

Keep the generic `DataCollectionExecutor` test at `origin/teacache_alex_rewrite@263ae466:tests/diffusion/cache/test_teacache_unit.py:335` to `349`. After the Hunyuan tiny boundary test passes, run a Hunyuan collection job with that executor around `self.layers`, not around `ragged_final_layer` or the final diffusion prediction. The target collector records `modulated_input` and the first computed model output at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/coefficient_estimator.py:21` to `46`, calculates relative L1 differences at lines 140 to 144, and fits a fourth order polynomial at lines 147 to 168. The paper defines the relative L1 input difference in Eq. 4 on PDF page 4 and the polynomial rescaling in Eq. 6 and Eq. 7 on PDF page 5.

For the Hunyuan collection run, record the prompt, image size, seed, timestep schedule, guidance mode, SP size, dtype, boundary shape, and coefficient tuple. Check that the fitted result has five finite values and that replaying the collected trajectory produces the expected hit and miss sequence. The paper does not specify Hunyuan prompts or a Hunyuan threshold, so the collection data and quality checks are required evidence rather than a paper supplied constant. Do not reuse the old Hunyuan coefficient tuple as proof. The old tuple is attached to the old protocol stub at [hunyuan_image3_transformer.py:2378](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:2378), while the old loop caches the final prediction at [hunyuan_image3_transformer.py:3172](/home/aja/vllm-omni/vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:3172).

### Finish with real image quality and performance tests

Extend the existing Hunyuan accuracy test with a cache comparison after all structural tests pass. The file is marked `local_model` and `diffusion` at [test_hunyuan_image3.py:76](/home/aja/vllm-omni/tests/e2e/accuracy/test_hunyuan_image3.py:76), imports image similarity helpers at lines 22 to 28, and already compares generated images with SSIM and PSNR in the offline image test at lines 524 to 567. Run the no cache and TeaCache cases with the same real model, prompt, seed, size, guidance scale, and number of steps. Compare image quality with the existing helpers and record the number of computed layer passes and elapsed time.

Use `--run-level advanced_model` for this real weight test because the repository uses `core_model` to build tiny models and `advanced_model` or `full_model` to load real weights at [tests/model_tests/conftest.py:9](/home/aja/vllm-omni/tests/model_tests/conftest.py:9) to [tests/model_tests/conftest.py:35](/home/aja/vllm-omni/tests/model_tests/conftest.py:35). A command after adding a TeaCache case is:

```text
HUNYUAN_MODEL_PATH=/path/to/model \
.venv/bin/pytest --run-level advanced_model -q \
  tests/e2e/accuracy/test_hunyuan_image3.py -k teacache
```

Do not set a fixed Hunyuan speedup or image metric threshold from the TeaCache paper. The paper reports results for its tested video models on PDF pages 6 to 8, not for Hunyuan Image 3. Choose Hunyuan thresholds from the no cache baseline and record them with the measured outputs.

## Migration acceptance checklist

The port is ready for broader benchmarking only when all of the following have evidence:

* The target CPU TeaCache unit suite passes, including the new shape change case.
* `HunyuanImage3Model` satisfies the native protocol and the generic backend installs and refreshes its runtime.
* The CPU Hunyuan boundary test proves first step, later miss, later hit, mode guards, residual reuse, and shape policy.
* The one GPU tiny model test proves that real Hunyuan decoder layers are skipped on a hit.
* Existing ImageKVCacheManager CPU tests still pass, including CFG and SP cases.
* The two GPU SP test proves equal rank decisions, no hang, correct gather shape, and no cache output parity.
* The separate CFG parallel test proves positive and negative residual isolation.
* The lifecycle test proves refresh clears state between normal generations.
* Hunyuan coefficients are collected at the new boundary and are finite.
* The real weight test passes image quality checks against the no cache baseline.
* Speed and memory are measured only after the correctness checks pass.

The codebase and the paper support the test boundaries and the residual definition. They do not establish Hunyuan image quality, speedup, memory use, or distributed TeaCache safety. Those claims require the tests above to run.

## Claims not supported by the current evidence

The following claims must not appear in the implementation plan as facts until measured:

* Hunyuan Image 3 will achieve a specific speedup. The TeaCache paper reports speedups for its tested video models on PDF pages 6 to 8, not for Hunyuan Image 3.
* Hunyuan Image 3 will use a specific amount of memory. The runtime stores detached residual tensors at `origin/teacache_alex_rewrite@263ae466:vllm_omni/diffusion/cache/teacache/runtime.py:121`, so the amount depends on the Hunyuan boundary tensor shape, dtype, sharding, and branch state.
* The existing Hunyuan coefficient tuple will preserve image quality after the boundary changes. The estimator supports fitting against the chosen boundary, but the current code does not contain a Hunyuan fit for the new boundary.
* SSIM or PSNR thresholds from the paper are valid Hunyuan acceptance criteria. The paper reports those metrics for its video experiments on PDF page 7. It does not define Hunyuan Image 3 thresholds.
* Step execution is safe with one model scoped runtime. The current step code groups request states, while the runtime has one state object without a request key.

## Minimal implementation order

The evidence supports this order:

1. Add the new native attributes and remove the old Hunyuan protocol stubs.
2. Move the existing layer loop into a local `compute_fn` inside `HunyuanImage3Model.forward`.
3. Add the executor call around that function for later `gen_image` steps only.
4. Pass the existing timestep embedding as `modulated_input`.
5. Make `do_true_cfg` explicit for sequential and CFG parallel execution.
6. Replace the Hunyuan custom backend enabler with the generic native backend path.
7. Keep step execution cache rejection.
8. Collect new coefficient data at the native boundary.
9. Run unit tests, then Hunyuan model tests, then advanced model accuracy tests, then measure speed and memory.

The first six steps follow the refactor interface and the current Hunyuan call graph. The coefficient, accuracy, speed, and memory results require execution and are not established by the paper alone.
