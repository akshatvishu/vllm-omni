# Hunyuan Image 3: Main Versus TeaCache Refactor Branch

## Scope

This report compares `main` at `a874b8e09` with the Hunyuan Image 3 implementation in the `teacache_rewrite_base` worktree, including its current uncommitted changes. It covers the TeaCache refactor and the other Hunyuan-specific changes in the worktree.

## Conclusion

On `main`, Hunyuan TeaCache lives in the outer diffusion pipeline. A cache hit skips the complete model call and reuses the previous final diffusion prediction. This branch moves the cache boundary into `HunyuanImage3Model`, around the decoder layers only. The pipeline still prepares inputs, advances Hunyuan KV state, runs the model, and updates generation state on every step. On a cache hit, only the expensive decoder-layer computation is reconstructed from a cached residual; normal final image postprocessing remains at the output stage.

That boundary is the important change. Hunyuan has input preparation, image KV management, sequence-parallel layout, CFG handling, and output processing around the decoder layers. Skipping the complete model call can skip state updates that are not part of the cached final prediction.

## 1. Workflow difference

### `main`: pipeline-level final-prediction reuse

```text
TeaCacheBackend.enable
        |
        v
Hunyuan custom enabler
        |
        v
pipeline._tea_cache_config

Each denoise step
        |
        v
time_embed(timestep) -> relative L1 -> polynomial threshold
        |
   +----+----+
   |         |
  miss      hit
   |         |
   v         v
full        skip model.forward_call
model call  reuse tc_prev_pred
   |         |
   +----+----+
        |
        v
scheduler / next step
```

The old cache stored the final `diffusion_prediction`. A hit skipped `model.forward_call`, and model kwargs were updated only after a miss. The backend therefore needed a Hunyuan-specific enabler and a refresh special case.

### This branch: model-level decoder-layer residual reuse

```text
TeaCacheBackend.enable
        |
        v
supports_teacache(model)
        |
        v
TeaCacheRuntime installed on HunyuanImage3Model

Each denoise step
        |
        v
prepare image/text inputs, KV state, padding, and SP layout
        |
        v
HunyuanImage3Model.forward
        |
        v
explicit cache guards
        |
   +----+----+
   |         |
  miss      hit
   |         |
   v         v
self.layers  current hidden states
run normally  + cached residual
   |         |
   +----+----+
        |
        v
remaining model path, CFG, output projection, scheduler
```

The runtime stores `layer_output - layer_input` at the decoder-layer boundary. On a hit it returns the current layer input plus that residual. The model remains responsible for deciding whether the boundary is safe to cache.

### The new cache boundary

```text
Hunyuan input preparation and KV state
                    |
                    v
              hidden_states
                    |
          TeaCache boundary
              self.layers only
                    |
                    v
      SP gather, final layer, CFG, output
                    |
                    v
                scheduler
```

## 2. Why the boundary moved

| Concern | `main` | This branch | Reason |
| --- | --- | --- | --- |
| Cache unit | Final diffusion prediction | Decoder-layer residual | Keeps the cache close to the repeated transformer work and avoids reusing an output produced from stale surrounding state. |
| Cache owner | Hunyuan-specific pipeline code | Generic `TeaCacheRuntime` attached to a capable model | Removes the Hunyuan custom enabler and gives the backend one lifecycle and reset path. |
| State progression | A hit skips the complete model call | Input preparation and model call continue every step | Hunyuan KV, padding, SP metadata, CFG inputs, and output bookkeeping can advance on hits. |
| Safety checks | Pipeline threshold only | Model checks mode, first step, CFG prefill, optional outputs, `use_cache`, metric, and executor state | The model has the context needed to reject unsafe cache use. |
| CFG | Implicit in the outer loop | Explicit `tea_cache_do_true_cfg` flag and runtime branch state | Prevents conditional and unconditional paths from sharing the wrong residual. |
| KV cache | Whole-call reuse could bypass normal KV handling | Native TeaCache forces `use_cache=False`; Hunyuan's prompt KV manager remains active | The native runtime returns only the declared residual output and cannot preserve decoder KV outputs. |
| Lifecycle | Hunyuan refresh was a no-op around per-call state | Backend refresh resets installed runtimes | Runtime state is explicit and owned by the backend. |
| Coefficients | Global pipeline coefficient map | Model getter with a finite provisional fallback | Coefficients depend on the cache boundary. The old final-prediction calibration is not evidence for the new decoder-layer boundary. |

## 3. Hunyuan changes in this branch

### TeaCache-specific changes

- `HunyuanImage3Model` now advertises native TeaCache support through `supports_teacache`, `tea_cache_model_key`, `tea_cache_executor`, and `get_teacache_coefficients()`.
- The decoder-layer loop is isolated in a closure and passed to `TeaCacheBlockExecutor`. The cacheable path is limited to image generation after the first step, without unconditional CFG prefill, optional attention/hidden-state outputs, or `use_cache`.
- The model passes the time embedding as the TeaCache modulation input and forwards the explicit true-CFG flag.
- `pipeline_hunyuan_image3.py` forwards the metric and CFG flag and preserves `use_cache` in generation kwargs.
- The outer Hunyuan pipeline no longer owns `_tea_cache_config`, `tc_rescale`, `tc_prev_pred`, or the final-prediction skip loop. It always prepares inputs and calls `model.forward_call`; the native executor disables decoder `use_cache` when needed, and kwargs are updated on every non-final step.
- The image rotary embedding output is restored to `[tokens, heads, head_dim]` and converted to `bfloat16`, matching the native Hunyuan image-attention and KV-cache layout.

### Adjacent Hunyuan execution changes

These changes are in the branch's Hunyuan implementation but are separate from the TeaCache algorithm:

- Added `hunyuan_fused_moe.py` and exported `HunyuanFusedMoE`. It adapts the upstream fused-MoE factory to Hunyuan's `ForwardContext`, one-time kernel initialization, expert parameter mapping, and omni-platform dispatch. This is needed for constructing and running the native Hunyuan layers on the current stack.
- Replaced the generic fused-MoE use in Hunyuan with `HunyuanFusedMoE`, removed the obsolete diffusion fused-MoE context reset, and added a local `repeat_kv` implementation.
- Updated the Cache-DiT adapter import to the current `cache_dit_backend` module and removed no-longer-used generic all-gather and platform helpers.
- Simplified image KV handling by removing the unused `allgather_size` and old Ulysses/all-gather test patching, and always expanding K/V heads into the native attention layout.
- Updated request-state typing from `StepRequestState` to `DiffusionRequestState`.
- Moved Hunyuan image postprocessing into `post_decode` and returned `DiffusionOutput` with `custom_output`, including the latent-output path, instead of using the removed Hunyuan-specific engine postprocess callback.
- Added native fused-MoE coverage and updated the Hunyuan step-execution and image-KV tests for those interface and layout changes.
- Added the fast CPU and native tiny-model TeaCache tests described below.

There is also a comment-only spelling change from `use` to `useing` in the Hunyuan tokenizer diff. It has no runtime effect and should not be retained as part of the feature.

## 4. Tiny-model and CPU test strategy

The tests deliberately use two layers of coverage:

```text
Fast CPU control-flow test
HunyuanImage3Model + CountingLayer stubs
        |
        +--> first step bypass
        +--> later-step miss
        +--> repeated-input hit
        +--> negative paths bypass

Native tiny-model test
HunyuanImage3Model + actual HunyuanImage3DecoderLayer instances
small config + seeded weights + one-rank CPU patches
        |
        +--> actual Hunyuan attention/KV/layer boundary
        +--> first step, miss, hit
        +--> shape transition miss
        +--> changed-metric forced miss
```

The fast test isolates TeaCache control flow and remains cheap. The native test is model-native rather than a generic mock: it constructs actual `HunyuanImage3DecoderLayer` objects with a reduced Hunyuan configuration and deterministic seeded parameters. It validates the integration point without loading a full checkpoint, VAE, tokenizer, or distributed runtime.

Observed results in this worktree:

- Hunyuan Image 3 test directory: `220 passed, 1 skipped`.
- Focused TeaCache, Hunyuan, KV-cache, step-execution, and backend tests: `46 passed`.
- Native Hunyuan TeaCache selection: `2 passed`.
- The complete TeaCache protocol file reached `29 passed`; four Flux2/Flux2-Klein cases failed before assertions because the 5.66 GiB GPU was out of memory. Those failures were not Hunyuan failures.

## 5. Evidence and remaining limits

The implementation and tests establish the control flow and the native Hunyuan layer boundary. They do not yet establish production cache quality or speed:

- The Hunyuan coefficient tuple is explicitly a finite fallback. It was not collected for the new decoder-layer boundary.
- `DataCollectionExecutor` exists for native-boundary collection, but the general coefficient estimator still uses its legacy collection-hook path and has no Hunyuan adapter. A dedicated collection run and Hunyuan adapter are still needed before treating the coefficients as calibrated.
- The tests use CPU and one-rank patches. Distributed sequence parallelism, true CFG parallel execution, full checkpoint output quality, and end-to-end speedup remain unvalidated here.

The branch's main design choice is therefore justified for integration correctness: preserve Hunyuan's surrounding stateful workflow and cache only the native decoder-layer computation. Coefficient calibration and hardware-level performance validation are follow-up evidence, not reasons to move the cache back to the outer pipeline.
