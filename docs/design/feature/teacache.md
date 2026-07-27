# TeaCache

TeaCache skips a diffusion transformer's block region when consecutive denoising
steps have similar modulated inputs. The model owns the boundary because only the
model knows which preprocessing, blocks, and output steps must stay together.

## Model contract

A native TeaCache model declares the following class attributes and method:

```python
supports_teacache = True
tea_cache_model_key = "QwenImageTransformer2DModel"
tea_cache_executor: TeaCacheBlockExecutor | None = None

def get_teacache_coefficients(self) -> list[float]:
    return [a4, a3, a2, a1, a0]
```

The model keeps its normal `forward()` path when the executor is `None`. When the
executor is installed, the model computes the metric at the first block boundary
and passes the existing block loop to `TeaCacheRuntime`:

```python
def run_transformer_blocks() -> tuple[torch.Tensor, ...]:
    # Keep the model's existing block order and arguments here.
    ...

hidden_states, encoder_hidden_states = executor.run(
    modulated_input=modulated_input,
    residual_inputs=(hidden_states, encoder_hidden_states),
    compute_fn=run_transformer_blocks,
    do_true_cfg=self.do_true_cfg,
)
```

`residual_inputs` names the tensors at the block boundary. The compute function
must return one output for each residual input, and every output must have the same
shape as its input. Single stream models pass one tensor. Dual stream models pass
both image and encoder tensors when the current implementation caches both.

The `supports_teacache()` helper checks the capability flag, model key, executor
slot, and coefficient method before the backend installs the runtime. The model key
is separate from the Python class name so two models with the same class name can
still select different cache behavior.

## Runtime behavior

`TeaCacheRuntime` owns the mutable state for one installed model. It keeps a shared
forward counter and separate positive and negative branch states. It preserves the
existing branch rules:

- Without true CFG, the positive state is used.
- CFG parallel rank zero uses the positive state and other ranks use the negative state.
- Sequential true CFG alternates positive and negative states by forward count.

The runtime always computes the first call for a branch. Later calls compute the
relative L1 distance, apply the model's fourth order polynomial, accumulate the
absolute rescaled distance, and compare it with `rel_l1_thresh`. A full compute
resets the accumulator. A cache hit adds the last detached output minus input
residual to the current boundary tensors.

The runtime clones the declared residual inputs before calling the block function,
because a block may update its inputs in place. It keeps the residuals on the same
device, with the same dtype, shape, and partitioning as the model outputs. A
nonfinite distance forces a full compute so invalid values do not keep stale cache
state alive.

Call `TeaCacheBackend.refresh()` before each new generation. Refresh resets every
runtime installed by that backend, including explicit Bagel and SenseNova targets.

## Backend target selection

The backend installs the runtime on `pipeline.transformer` for the standard native
targets. Bagel uses `pipeline.bagel` and keeps the existing pipeline transformer
alias. SenseNova uses the nested `SenseNovaU1ForCausalLM` selected by its denoising
adapter. The adapter remains the CFG owner, so prefix and understanding forwards do
not enter the denoising cache boundary. HunyuanImage3 keeps its pipeline-local
TeaCache state and does not use a transformer executor.

The current native targets are:

| Model key | Cached boundary tensors |
| --- | --- |
| `QwenImageTransformer2DModel` | Image and encoder hidden states |
| `Bagel` | Packed hidden sequence |
| `ZImageTransformer2DModel` | Unified hidden sequence |
| `Flux2Klein` | Final image hidden state |
| `StableAudioDiTModel` | Full hidden sequence, including the global token |
| `Flux2Transformer2DModel` | Final image hidden state and the unchanged encoder tensor |
| `LongCatImageTransformer2DModel` | Image and encoder hidden states |
| `FluxTransformer2DModel` | Image and encoder hidden states |
| `SenseNovaU1ForCausalLM` | Denoising hidden state |

Flux2 keeps its unchanged encoder tensor in the residual tuple because the legacy
path stored a zero encoder residual. The native path keeps that behavior until a
separate parity test proves that removing it has no output, state, memory, or
performance effect.

## Coefficients

The backend resolves coefficients in this order:

1. `DiffusionCacheConfig.coefficients` when the user provides an override.
2. `get_teacache_coefficients()` on the selected model.
3. The fixed HunyuanImage3 coefficients in the pipeline-native enabler.

The resolved `TeaCacheConfig` is immutable. It requires exactly five finite
coefficients and a finite positive threshold.

The coefficient estimator installs a native data collection executor at the same
model boundary used in production. It records the metric and the first block output
for each denoising call, so the collected run uses the same boundary as production.

## Testing

CPU tests cover capability validation, configuration validation, cache-hit and
cache-miss decisions, residual arity and shape checks, CFG state isolation, reset,
and nonfinite metric handling. The native model contract test checks every migrated
model class without requiring a checkpoint.

The GPU integration test uses a small Qwen-Image transformer with finite test
weights. It runs on one CUDA device, installs `TeaCacheRuntime`, counts calls to the
real transformer blocks, and verifies that an identical second denoising call reuses
the cached residual. The test is marked for the single H100 resource class used by
the GPU test suite. A full Qwen-Image checkpoint fits within the 192 GB device
budget, while the tiny test model keeps the test time and memory use bounded.

## Adding support for a model

Use the following sequence when adding a model:

1. Find the existing block boundary in the model's `forward()` method.
2. Add the capability attributes and model coefficient method.
3. Keep preprocessing and postprocessing in `forward()`, and wrap only the block region.
4. Declare every residual tensor that the current implementation caches.
5. Add a cache-disabled test and a test that proves the executor skips the block region on a hit.
6. Add an explicit backend enabler only when the pipeline target is not `pipeline.transformer`.

TeaCache is not supported with another diffusion cache backend at the same time,
Pipeline Parallel, step execution, or diffusion continuous batching.
