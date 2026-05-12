# GGUF Quantization

## Overview

GGUF support loads pre-quantized diffusion transformer weights while keeping
the rest of the pipeline on the base Hugging Face checkpoint. Use the base
model for tokenizer, text encoder, scheduler, and VAE, then pass the GGUF file
for the transformer.

GGUF is static quantization: the quantized weights are produced before serving.

## Hardware Support

| Device | Support |
|--------|---------|
| NVIDIA Blackwell GPU (SM 100+) | ✅ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ |
| NVIDIA Ampere GPU (SM 80+) | ✅ |
| AMD ROCm | ⭕ |
| Intel XPU | ⭕ |
| Ascend NPU | ❌ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this
guide.

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

| Model | HF base model | GGUF input | Scope | Adapter |
|-------|---------------|------------|-------|---------|
| Qwen-Image family | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512`, edit and layered Qwen-Image pipelines | Local `.gguf`, `repo/file.gguf`, or `repo:quant_type` | Transformer only | `QwenImageGGUFAdapter` |
| Wan2.2 | Wan2.2 diffusion pipelines | Local `.gguf`, nested HF `.gguf`, or per-source `gguf_models` | Transformer only | `Wan22GGUFAdapter` |
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | Local `.gguf`, `repo/file.gguf`, or `repo:quant_type` | Transformer only | `ZImageGGUFAdapter` |
| FLUX.2-klein | `black-forest-labs/FLUX.2-klein-4B` | Local `.gguf`, `repo/file.gguf`, or `repo:quant_type` | Transformer only | `Flux2KleinGGUFAdapter` |

Generic FLUX.1 GGUF checkpoints are not listed here; the implemented adapter is
for the FLUX.2-klein path.

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| Qwen3-Omni | Thinker language-model stage | Not validated | GGUF is not documented for omni/TTS AR stages |
| Qwen3-TTS | TTS language-model stage | Not validated | GGUF is not documented for TTS stages |

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| BAGEL | Stage-specific transformer weights | Not validated | Requires a model-specific GGUF adapter |
| GLM-Image | Stage-specific transformer weights | Not validated | Requires a model-specific GGUF adapter |

## Configuration

Offline:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --gguf-model QuantStack/Qwen-Image-GGUF/Qwen_Image-Q4_K_M.gguf \
  --quantization gguf \
  --prompt "a red paper kite hanging from a pine tree in a winter courtyard" \
  --height 1024 \
  --width 1024 \
  --seed 42 \
  --num_inference_steps 20 \
  --output outputs/qwen_image_q4km.png
```

Online:

```bash
vllm serve Qwen/Qwen-Image \
  --omni \
  --port 8000 \
  --quantization-config '{"method":"gguf","gguf_model":"QuantStack/Qwen-Image-GGUF/Qwen_Image-Q4_K_M.gguf"}'
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | - | Quantization method (`"gguf"`) |
| `gguf_model` | str | - | Local GGUF file, explicit Hugging Face file, or `repo:quant_type` selector |
| `gguf_models` | dict[str, str] | - | Per-source GGUF refs keyed by component subfolder, such as `transformer` and `transformer_2` |

`gguf_model` accepts the following forms. Each `gguf_models` value accepts the
same forms.

| Form | Example |
|------|---------|
| Local file | `/models/z-image-Q4_K_M.gguf` |
| Explicit HF file | `QuantStack/Qwen-Image-GGUF/Qwen_Image-Q4_K_M.gguf` |
| Nested HF file | `QuantStack/Wan2.2-T2V-A14B-GGUF/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf` |
| HF repo plus quant type | `owner/repo:Q4_K_M` |

Multi-transformer models can use `gguf_models` to route each transformer
source to a separate GGUF file:

```json
{
  "method": "gguf",
  "gguf_models": {
    "transformer": "QuantStack/Wan2.2-T2V-A14B-GGUF/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
    "transformer_2": "QuantStack/Wan2.2-T2V-A14B-GGUF/LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"
  }
}
```

## Wan2.2 GGUF

Wan2.2 GGUF repositories can store each diffusion transformer in a separate
subdirectory. For example, `QuantStack/Wan2.2-T2V-A14B-GGUF` stores the
high-noise transformer under `HighNoise/` and the low-noise transformer under
`LowNoise/`.

Single-transformer Wan2.2 variants can use `gguf_model`:

```bash
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --omni \
  --port 8000 \
  --quantization-config '{"method":"gguf","gguf_model":"QuantStack/Wan2.2-TI2V-5B-GGUF/Wan2.2-TI2V-5B-Q4_K_M.gguf"}'
```

Two-transformer Wan2.2 variants should use `gguf_models`:

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni \
  --port 8000 \
  --quantization-config '{"method":"gguf","gguf_models":{"transformer":"QuantStack/Wan2.2-T2V-A14B-GGUF/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf","transformer_2":"QuantStack/Wan2.2-T2V-A14B-GGUF/LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"}}'
```

The base model still supplies the scheduler, tokenizer, text encoder, and VAE.
Only transformer weights are routed through GGUF.

## Validation and Notes

1. `OmniDiffusionConfig` receives `{"method": "gguf", "gguf_model": ...}` or
   `{"method": "gguf", "gguf_models": ...}`.
2. `DiffusersPipelineLoader` resolves the GGUF file.
3. A model-specific adapter remaps GGUF tensor names to vLLM-Omni transformer
   names.
4. Only transformer weights are loaded from GGUF. Missing non-transformer
   weights are loaded from the base model repository.
5. vLLM's GGUF linear method performs dequantization and GEMM at runtime.

Unsupported models fail fast with a clear "No GGUF adapter matched" error
instead of falling back to a generic mapper. Many GGUF repositories do not
include `model_index.json`; always pass the normal base model through `--model`.
