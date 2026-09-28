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

HunyuanImage3's legacy TeaCache path does not support configurable warmup. A nonzero `num_warmup_steps` raises an error instead of being silently ignored.

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

Cosmos3 Nano and Super use the coefficients proposed in [PR #4389](https://github.com/vllm-project/vllm-omni/pull/4389). That PR fitted them on Cosmos3 Nano text-to-video runs and recommended 12 warmup steps to reduce early-step quality loss. The protocol path includes the final GEN norm in the cached residual, while #4389 fitted a residual before that norm. On one MI300X with four prompt and seed pairs, the measured speedups were comparable (1.52× for #4389 and 1.55× for the protocol path). The protocol path had less measured output change against its own uncached reference. The runs used different vLLM versions (0.24.0 and 0.30.0), so the comparison does not isolate the rewrite's effect. This check covers Nano text-to-video only. Transfer requests run without TeaCache because their branches have different GEN layouts. Cosmos3 Edge has no calibrated default coefficients and requires an explicit coefficient override. Edge also defaults to zero warmup steps; set `num_warmup_steps` explicitly when supplying Edge coefficients.

Cosmos3 TeaCache cannot be combined with HSDP, layerwise offload, or distributed layerwise offload. The current TeaCache signal reads block zero's GEN weights before its forward. Layerwise offload loads that block synchronously at setup, then prefetches it asynchronously on a separate stream between runs. The early TeaCache read has not been validated with that prefetch. HSDP and distributed layerwise offload also require a shared skip decision across weight-sharding groups.

Bagel uses its decomposed forward with the existing coefficients and default threshold of 0.2. The legacy TeaCache extractor ignored CFG arguments and ran only the conditional branch. The protocol path restores CFG, so earlier measurements of the legacy path do not establish quality or speed for this path. Batched CFG branches occupy separate rows in the cached residual. Calls outside the CFG interval run without caching, because they use a different packed layout. The port preserves prefix KV caches and applies CFG after the cached blocks. CPU tests cover these paths; output quality and speed with real weights remain unverified. Generation paths that call `forward_single_branch`, including the existing SP and CFG-parallel paths, bypass TeaCache.

SenseNova U1's default threshold of 0.2 is not quality validated. A one-MI300X comparison at 768×768, 50 steps and CFG scale 4 found mean SSIM 0.498 and LPIPS 0.443 against uncached output across ten prompt and seed pairs, with visible blur in one inspected image. A diagnostic request reused the cache in 90 of 100 branch calls. The inherited coefficients or threshold need further calibration before recommending this setting. U1.5 has not been checked with real weights.

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
