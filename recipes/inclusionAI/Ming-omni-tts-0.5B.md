# Ming-omni-tts 0.5B

## Summary

- Vendor: inclusionAI
- Model: `inclusionAI/Ming-omni-tts-0.5B`
- Deploy config: `vllm_omni/deploy/ming_tts.yaml`
- Pipeline: dense two-stage AR + Flow/VAE Ming TTS
- Output: mono 44.1 kHz audio

## Examples

- Offline: [`examples/offline_inference/text_to_speech/ming_tts/`](../../examples/offline_inference/text_to_speech/ming_tts/)
- Online: [`examples/online_serving/text_to_speech/ming_tts/`](../../examples/online_serving/text_to_speech/ming_tts/)

## Install

```bash
export VLLM_VERSION="0.21.0"
uv venv
source .venv/bin/activate
uv pip install vllm==$VLLM_VERSION --torch-backend=cu130
uv pip install -e .
uv pip install soundfile pyyaml openai aiohttp huggingface_hub
```

## Offline

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
  --model inclusionAI/Ming-omni-tts-0.5B \
  --case style \
  --deploy-config vllm_omni/deploy/ming_tts.yaml \
  --enforce-eager
```

The offline example owns the full case list, including speech, zero-shot,
podcast, text-to-audio, and music-style workflows.

## Online

```bash
vllm-omni serve inclusionAI/Ming-omni-tts-0.5B \
  --deploy-config vllm_omni/deploy/ming_tts.yaml \
  --omni \
  --port 8091 \
  --enforce-eager
```

```bash
python examples/online_serving/text_to_speech/ming_tts/openai_speech_client.py \
  --text "你好，这是 Ming 在线语音合成测试。" \
  --max-new-tokens 200
```

## Hardware

Validated on NVIDIA A100 40GB and L4 class GPUs. Local CPU-only environments
are suitable for static checks, but functional Ming generation requires CUDA.
