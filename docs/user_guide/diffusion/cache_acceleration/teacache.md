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

TeaCache can reduce diffusion inference time by reusing transformer block residuals across denoising steps. A residual is the difference between the block output and its input. TeaCache compares a model-provided input signal between steps to decide when to run the blocks again.

Caching can change the generated output. Speed and output quality depend on the model, threshold and generation settings; compare cached and uncached output for your workload.

See [Supported Models](../../diffusion_features.md#supported-models) for availability and [Model Notes](#model-notes) for restrictions.

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

Pass `--cache-backend tea_cache` to the text-to-image or image-editing example:

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

Set these options in `cache_config` for offline inference or `--cache-config` for serving.

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `rel_l1_thresh` | float \| None | Model default, usually `0.2` | Positive threshold for accumulated, polynomial-rescaled input change. Lower values make reuse more conservative. MiniMax-H3 defaults to `0.17`. |
| `coefficients` | list[float] \| None | `None` | Polynomial coefficients for rescaling L1 distance. Must contain exactly 5 elements if provided. If `None`, uses model-specific defaults based on transformer type. |
| `num_warmup_steps` | int \| None | Model default | Nonnegative number of initial cache-eligible calls per cache state that always run the blocks. Defaults to 12 for Cosmos3 Nano and Super, and 0 otherwise. |

HunyuanImage3 does not support `num_warmup_steps`; a nonzero value raises an error.

Unspecified settings use the model defaults. Override coefficients only with values validated for the model and cached block range.

### Model Notes

#### Cosmos3

- **Defaults:** Nano and Super use threshold 0.2 and 12 warmup steps. Edge requires explicit `coefficients`; its warmup count defaults to 0.
- **Measured:** On one MI300X, Nano text-to-video (1280×720, 189 frames, 35 steps, CFG 6, threshold 0.2, four prompt and seed pairs) reduced mean API request time from 250.1 to 161.3 seconds, a 1.55× speedup. Mean SSIM was 0.866 and LPIPS was 0.052 against matching uncached outputs. Super, Edge, image generation and distributed modes have not been measured.
- **Restrictions:** HSDP, layerwise offload and distributed layerwise offload raise an error. Transfer requests run without TeaCache.

#### Bagel

- **Defaults:** Threshold 0.2 with the existing coefficients.
- **Restrictions:** With SP or CFG-parallel, Bagel runs without TeaCache. Calls outside the CFG interval also run without caching.
- **Measured:** On one MI300X, text-to-image at 512×512 with 50 requested timesteps and text CFG 4 reduced median request time from 5.904 to 1.931 seconds. Mean SSIM was 0.745 and LPIPS was 0.195 across ten prompt and seed pairs repeated three times. Both modes used the corrected CFG implementation. Cached images changed composition and detail; the run predates the subsequent CFG state cleanup.

#### SenseNova-U1

- **Defaults:** Threshold 0.2 with the coefficients from [PR #4164](https://github.com/vllm-project/vllm-omni/pull/4164).
- **Measured:** On one MI300X (768×768, 50 steps, CFG 4, ten prompt and seed pairs), the default gave mean SSIM 0.498 and LPIPS 0.443 against uncached output, with visible blur in one inspected image. The cache was reused in 90 of 100 branch calls in one request. Try a lower threshold and compare outputs again, or disable caching if the changes are unacceptable. U1.5 has not been measured.
- **Restrictions:** Requests without CFG run without TeaCache. Three-branch image editing with `cfg_parallel_size=2` raises an error.

---

## Best Practices

Compare with `cache_backend="none"` or `--cache-backend none`, using the same checkpoint, prompts, seeds, resolution, inference steps, CFG, dtype and offload settings. Warm up both configurations before measuring request latency and denoising time.

SSIM and LPIPS measure similarity to uncached output. They do not establish an acceptable quality level for every workload, so inspect the saved outputs as well. If exact agreement is required, run without caching.

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

**Symptoms**: Caching produces few hits or little reduction in request latency.

**Solutions**:

1. Enable `VLLM_LOGGING_LEVEL=DEBUG` to inspect `TeaCache step=... cache_hit=...` messages from `TeaCacheHook`. An enablement message only confirms that the hook was installed. HunyuanImage3 uses a separate native loop.
2. Check [Model Notes](#model-notes) for calls that run uncached. Warmup calls also execute the blocks.
3. Measure denoising time separately from text encoding, VAE work and request overhead. Increasing the threshold permits more reuse, but requires another output comparison.

---

## Summary

Enable TeaCache with `cache_backend="tea_cache"`, check the model restrictions and compare against uncached generation before choosing a threshold.
