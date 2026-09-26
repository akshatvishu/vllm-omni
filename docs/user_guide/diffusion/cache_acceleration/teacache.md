# TeaCache Guide

## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

TeaCache accelerates diffusion model inference by caching transformer computations when consecutive timesteps are similar, providing **1.5x-2.0x speedup** with minimal quality loss. It dynamically decides whether to reuse cached outputs based on input similarity, making it ideal for production deployments where inference speed matters without sacrificing generation quality.

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
        "rel_l1_thresh": 0.2,  # Controls speed/quality tradeoff
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
| ----------- | ------ | --------- | ------------- |
| `rel_l1_thresh` | float | `0.2` | Similarity threshold for cache reuse. Lower values prioritize quality (less caching), higher values prioritize speed (more caching). Suggested range: 0.1-0.8 |
| `coefficients` | list[float] \| None | `None` | Polynomial coefficients for rescaling L1 distance. Must contain exactly 5 elements if provided. If `None`, uses model-specific defaults based on transformer type. |
| `num_warmup_steps` | int \| None | Model default | Initial denoising steps per CFG branch that always run the transformer. The default is 12 for Cosmos3 Nano and Super and 0 for other models. |

Ported models define their defaults in `get_teacache_defaults()`. Legacy model defaults remain in [`vllm_omni/diffusion/cache/teacache/config.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/cache/teacache/config.py), for example:

```python
_MODEL_COEFFICIENTS = {
    # Qwen-Image transformer coefficients from ComfyUI-TeaCache
    # Tuned specifically for Qwen's dual-stream transformer architecture
    # Used for all Qwen-Image Family pipelines, in general
    "QwenImageTransformer2DModel": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    ...
}
```

Cosmos3 Nano and Super use the coefficients proposed in [PR #4389](https://github.com/vllm-project/vllm-omni/pull/4389). That PR fitted the coefficients on Cosmos3 Nano text-to-video runs and recommended 12 warmup steps to reduce early-step quality loss. The protocol path includes the final GEN norm in the cached residual, while #4389 fitted a residual before that norm; output quality and speedup with these coefficients still need a real-weight check. Transfer requests run without TeaCache because their branches have different GEN layouts. Cosmos3 Edge has no calibrated default coefficients and requires an explicit coefficient override. Edge also defaults to zero warmup steps; set `num_warmup_steps` explicitly when supplying Edge coefficients.

Cosmos3 TeaCache cannot be combined with HSDP, layerwise offload, or distributed layerwise offload. The current TeaCache signal reads a GEN layer's weights before its forward. Layerwise offload prepares those weights in the block's forward hook, while HSDP and distributed layerwise offload also require a shared skip decision across weight-sharding groups.

---

## Best Practices

### When to Use

**Good for:**

- Production deployments requiring faster inference, tolerant of minimal quality loss
- Scenarios where 1.5-2x speedup is valuable
- Useful for single-card acceleration

**Not for:**

- Maximum quality requirements where no degradation is acceptable
- Very short inference runs (< 20 steps) where caching overhead may outweigh benefits

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

**Symptoms**: Actual speedup is less than expected (< 1.3x)

**Solutions**:

1. Increase the threshold to enable more aggressive caching:
   ```python
   cache_config={"rel_l1_thresh": 0.8}
   ```
2. Ensure you're using sufficient inference steps (35+ recommended)
3. Check that your model architecture is supported (see Supported Models section)

---

## Summary

1. ✅ **Enable TeaCache** - Set `cache_backend="tea_cache"` to get 1.5x-2.0x speedup with optimized defaults
2. ✅ **(Optional) Customize** - Adjust thresholds and polynomial coefficients for specific speed/quality trade-offs
