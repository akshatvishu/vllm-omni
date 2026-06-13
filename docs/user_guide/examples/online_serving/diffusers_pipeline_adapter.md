# Diffusers Backend Adapter

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/diffusers_pipeline_adapter>.


vLLM-Omni supports running diffusion models with the diffusers backend, directly serving any 🤗 Diffusers pipeline online without implementing them natively.

## Limitations

The diffusers backend is a black-box adapter. Its primary focus is to serve diffusion models online.
Currently, the following features are NOT yet supported.
It is not guaranteed whether they will be supported in the future.

- CFG parallel execution
- Sequence parallel execution
- TeaCache / Cache-DiT acceleration
- Step-wise execution (continuous batching)

For these features, it is recommended to use natively supported pipelines instead.

## Model Support

Any model loadable via `DiffusionPipeline.from_pretrained()` should run, including text-to-image, image-to-image, text-to-video, image-to-video, and text-to-audio.

However, as we strive to ensure output similarity between vLLM-Omni's diffuser backend and plain diffusers library, the following models are particularly verified:

- Qwen/Qwen-Image
- Tongyi-MAI/Z-Image-Turbo
- Wan2.2-I2V-A14B-Diffusers

If you find that a model not listed above also produces different outputs from running diffusers model directly.
Please consider file an issue.

## Usage

```bash
vllm serve "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --omni \
    --diffusion-load-format diffusers
```

Users turn on the diffusers backend primarily through the `--diffusion-load-format diffusers` argument.

### Single-File Checkpoints

For single-file checkpoints (such as `.safetensors` or `.ckpt`), users can load them via the `--diffusion-load-format diffusers_single_file` argument (or simply point `--model` to a local single checkpoint file).

If a Diffusers pipeline class is needed, specify it using `--model-class-name`:

```bash
vllm serve "/path/to/model.safetensors" \
    --omni \
    --diffusion-load-format diffusers_single_file \
    --model-class-name SomeDiffusersPipeline
```

Using `--diffusion-load-format diffusers_single_file` explicitly bypasses standard directory-based config loading. This allows you to pass a Hugging Face Hub ID (e.g. `repo/model`) or URL as the `--model` argument to fetch single files remotely, provided the specified Diffusers pipeline supports remote loading.

### Native Anima Single-File Checkpoints

Anima single-file checkpoints are served through the native `AnimaPipeline`, not through `AnimaModularPipeline.from_single_file()`. If `--model-class-name AnimaModularPipeline` is passed for a local single-file checkpoint, vLLM-Omni maps it to `AnimaPipeline`.

Use `--model-class-name AnimaPipeline`. The native path reads the Anima transformer single-file checkpoint directly, converts original Cosmos transformer keys when needed, and loads the Cosmos transformer and text conditioner into vLLM-Omni native modules.

The native path also needs the non-denoiser components (`text_encoder`, `tokenizer`, `t5_tokenizer`, `vae`, and optionally `scheduler`). These must be in Diffusers `from_pretrained()` layout. Raw Anima auxiliary files such as `qwen_3_06b_base.safetensors` and `qwen_image_vae.safetensors` are converter inputs; they are not accepted directly as `components_path`.

Use the Anima converter from the Diffusers reference implementation to prepare the component directory:

```bash
python /path/to/convert_anima_to_diffusers.py \
    --transformer_ckpt_path /path/to/anima-base-v1.0.safetensors \
    --text_encoder_ckpt_path /path/to/qwen_3_06b_base.safetensors \
    --vae_ckpt_path /path/to/qwen_image_vae.safetensors \
    --qwen_tokenizer_path /path/to/qwen-tokenizer \
    --t5_tokenizer_path /path/to/t5-tokenizer \
    --output_path /path/to/anima-components \
    --save_pipeline
```

Then point `--model` at the raw Anima transformer checkpoint and `components_path` at the converted directory:

```bash
vllm serve "/path/to/anima.safetensors" \
    --omni \
    --model-class-name AnimaPipeline \
    --diffusers-load-kwargs '{
      "components_path": "/path/to/anima-components"
    }'
```

No deploy config is required for local Anima single-file checkpoint discovery
when `--model-class-name AnimaPipeline` is provided.

Native Anima currently supports baseline single-GPU execution. Cache-DiT,
TeaCache, CPU offload, layer-wise offload, quantization, TP/SP, CFG parallel,
HSDP, and step execution are not supported by `AnimaPipeline` yet.

There are two more optional arguments, `--diffusers-load-kwargs` and `--diffusers-call-kwargs`, which are valid together with `--diffusion-load-format diffusers` or `diffusers_single_file`. Native Anima also accepts `--diffusers-load-kwargs` for component paths such as `components_path`, but does not delegate denoising to Diffusers.

After launching the model, users send a request as usual. Refer to other documentation pages on how to request a particular input/output modality, such as `examples/online_serving/text_to_image/openai_chat_client.py`.

## Configuration Reference

### `--diffusers-load-kwargs`

Passed as-is to `DiffusionPipeline.from_pretrained()`.

This is suitable for model-specific configurations not available through the vLLM-Omni interface.
For example: `--diffusers-load-kwargs '{"use_safetensors": true}'`.

When a parameter is available in the vLLM-Omni interface, it will be adapted here.
But if that parameter is simultaneously set in both the vLLM-Omni interface and `diffusers_load_kwargs`, the **latter** will take precedence.

### `--diffusers-call-kwargs`

Passed to `pipeline.__call__()`.

This is suitable for sampling parameters not available through the vLLM-Omni interface (such as online serving payloads).

When a parameter is available in the vLLM-Omni interface, it will be adapted here.
But if that parameter is simultaneously set in both the vLLM-Omni interface and `diffusers_call_kwargs`, the **former** will take precedence (because it is set at request time).

### Attention Backends

The diffusers backend converts
[vLLM-Omni standard of attention backend setting](../../../docs/user_guide/diffusion/attention_backends.md)
to [diffusers standard](https://huggingface.co/docs/diffusers/optimization/attention_backends#available-backends).

Specifically for `FLASH_ATTN`, it will first attempt to use FlashAttention-3 and then FlashAttention-2.

For each attempted version of `FLASH_ATTN` and `SAGE_ATTN`, it will first try to load the attention backend from HuggingFace `kernels` library, then without.

For unsuccessful attention selection or `TORCH_SDPA`, it will use the PyTorch's default attention backend.

The loaded attention backend and the failed attempts (if any) are logged to console.

### Model Specific Settings

The model loading and inference strictly follows the diffusers library, and they may be different from vLLM-Omni's native interface for some specific models.
Users are encouraged to double-check the model pipeline's interface in [diffusers' official documentation](https://huggingface.co/docs/diffusers/api/pipelines/overview).
Some particular examples are below.

#### Wan Series

The Wan series video generation models takes `boundary_ratio` and `flow_shift` during model initialization ([ref](https://huggingface.co/docs/diffusers/api/pipelines/wan)), not during inference.

Since our `OmniDiffusionConfig` contains these two values ([source](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/data.py)), we can directly pass `--boundary-ratio` and `--flow-shift` arguments to `vllm serve` command.

```bash
vllm serve "Wan2.2-T2V-A14B-Diffusers" \
    --omni \
    --boundary-ratio 0.875 \
    --flow-shift 3 \
    --diffusion-load-format diffusers
```

These extra CLI args will be attempted to pass as-is to the `OmniDiffusionConfig` dataclass and being accessible during model loading time.
Special routines inside the pipeline adapter ensures that they are set properly.
