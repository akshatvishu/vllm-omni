# TeaCache Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [Native Model Boundaries](#native-model-boundaries)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

TeaCache can skip a model-declared transformer block region when consecutive timestep embeddings are similar. It stores the detached residual from that region and applies it to the next boundary input. Speed, memory use, and output quality depend on the model, threshold, dtype, and boundary shape and must be measured for each model.

Native integrations expose the block boundary through `TeaCacheBlockExecutor`. Models that have not migrated still use the legacy hook integration. Hunyuan Image 3 uses the native boundary around its decoder layers and measures the time-conditioned image embedding returned by `patch_embed`.

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).

---

## Quick Start



### Basic Usage


```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

### Custom Configuration

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={
        "rel_l1_thresh": 0.2,
    },
)
```

### Using Environment Variable

You can also enable TeaCache via environment variable:

```bash
export DIFFUSION_CACHE_BACKEND=tea_cache
```

Then initialize without explicitly setting `cache_backend`:

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_config={"rel_l1_thresh": 0.2}
)
```

---

## Example Script

### Offline Inference

Use python script under `examples/offline_inference/text_to_image/` or `examples/offline_inference/image_to_image/` with CLI:

```bash
# Text-to-image example
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --cache-backend tea_cache

# Image-to-image example
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-Edit \
  --image input.png \
  --prompt "Edit description" \
  --cache-backend tea_cache \
  --tea-cache-rel-l1-thresh 0.25
```

See the [text_to_image.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/text_to_image/text_to_image.py) or [image_edit.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py) for detailed configuration options.

### Online Serving

```bash
# Default configuration
vllm serve Qwen/Qwen-Image --omni --port 8091 --cache-backend tea_cache

# Custom configuration
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh": 0.2}'
```

---

## Configuration Parameters

In `OmniDiffusionConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rel_l1_thresh` | float | `0.2` | Threshold for accumulated rescaled distance. Select it from an uncached baseline and a cached comparison. |
| `coefficients` | list[float] \| None | `None` | Polynomial coefficients for rescaling L1 distance. Must contain exactly 5 finite elements if provided. If `None`, the backend calls the selected transformer's coefficient getter. |

When `coefficients` is omitted, the backend asks the selected transformer for its coefficients. A custom list must contain exactly five finite values:

```python
cache_config={
    "rel_l1_thresh": 0.2,
    "coefficients": [a4, a3, a2, a1, a0],
}
```

---

## Native Model Boundaries

For Hunyuan Image 3, TeaCache is active only for later `gen_image` steps when `use_cache=False`. The first image step seeds the stable image-only shape and always executes the decoder layers. Text generation, unconditional CFG prefill, attention or hidden-state collection, `use_cache=True`, and a changed boundary shape also execute the layers.

The Hunyuan implementation caches decoder-layer residuals, not final diffusion predictions. The backend refreshes this state before a new generation. The current Hunyuan coefficient tuple is a provisional model default; it is not evidence of image quality or speed.

The CPU validation suite uses a tiny Hunyuan configuration and counting decoder layers:

```bash
./.venv/bin/pytest --run-level core_model -q \
  tests/diffusion/cache/test_teacache_unit.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_teacache.py
```

Use `--run-level advanced_model` only for real-weight Hunyuan quality and performance checks after the CPU boundary tests pass.

---

## Best Practices

### When to Use

**Good for:**

- Workloads where measured transformer-block reuse improves latency
- Experiments that compare cached and uncached outputs with the same seed
- Models with a stable, explicitly declared cache boundary

**Not for:**

- Maximum quality requirements where no degradation is acceptable
- Very short inference runs where cache overhead has not been measured


---

## Troubleshooting

### Common Issue 1: Quality Degradation

**Symptoms**: Generated images show artifacts, reduced detail, or inconsistent quality compared to non-cached results

**Solution**:

```python
# Lower the threshold for more conservative caching
cache_config={"rel_l1_thresh": 0.1}
```

### Common Issue 2: Limited Speedup

**Symptoms**: The measured latency improvement is smaller than expected

**Solutions**:
1. Increase the threshold to enable more aggressive caching:
   ```python
   cache_config={"rel_l1_thresh": 0.8}
   ```
2. Check that your model architecture is supported (see Supported Models section)

---


## Summary

1. ✅ **Enable TeaCache** - Set `cache_backend="tea_cache"` and validate the selected model's boundary
2. ✅ **(Optional) Customize** - Adjust thresholds and polynomial coefficients for specific speed/quality trade-offs
