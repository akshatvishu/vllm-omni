# Ming-omni-tts 0.5B

> Offline and online TTS/audio generation with the dense Ming two-stage AR + Flow/VAE pipeline

## Summary

- Vendor: inclusionAI
- Model: `inclusionAI/Ming-omni-tts-0.5B`
- Task: Text-to-speech, voice/style control, zero-shot cloning, podcast-style multi-speaker generation, and text-to-audio/music cases
- Mode: Offline `Omni` / `AsyncOmni` and online OpenAI-compatible `/v1/audio/speech`
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run the dense 0.5B Ming TTS model through
vLLM-Omni's two-stage pipeline:

- Stage 0: Qwen2-based autoregressive backbone with inline Ming flow controls
- Stage 1: audio VAE decode to mono 44.1 kHz waveform

The verified flow covers blocking offline generation, async-chunk offline
generation, and online serving for speech cases. Music-only `bgm` and `tta`
are covered by offline inference; the online `/v1/audio/speech` endpoint does
not yet expose the corresponding `prompt_mode` fields.

## References

- Hugging Face model:
  [`inclusionAI/Ming-omni-tts-0.5B`](https://huggingface.co/inclusionAI/Ming-omni-tts-0.5B)
- Offline example:
  [`examples/offline_inference/text_to_speech/ming_tts/`](../../examples/offline_inference/text_to_speech/ming_tts/)
- Online example:
  [`examples/online_serving/text_to_speech/ming_tts/`](../../examples/online_serving/text_to_speech/ming_tts/)
- Deploy config:
  [`vllm_omni/deploy/ming_tts.yaml`](../../vllm_omni/deploy/ming_tts.yaml)

## Installing vLLM-Omni

Use a fresh Python environment. The verified run used vLLM `0.21.0` with the
CUDA 13 PyTorch stack.

```bash
export VLLM_VERSION="0.21.0"

uv venv
source .venv/bin/activate
uv pip install vllm==$VLLM_VERSION --torch-backend=cu130
uv pip install -e .
uv pip install soundfile pyyaml openai aiohttp huggingface_hub
```

## Hardware Support

## GPU

### 1x A100 40GB

#### Environment

- OS: Linux
- Python: 3.12.13
- GPU: NVIDIA A100-SXM4-40GB, 40960 MiB
- Driver: 580.82.07
- PyTorch: `2.11.0+cu130`
- CUDA runtime reported by PyTorch: 13.0
- vLLM version: 0.21.0
- vLLM-Omni branch / commit: `feat/ming-omni-tts-dense` / `4d923c708099939178e932ff153c63749b430fd1`
- Deploy config: `vllm_omni/deploy/ming_tts.yaml`

#### Offline Command

Run a single blocking case:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
  --model inclusionAI/Ming-omni-tts-0.5B \
  --case style \
  --deploy-config vllm_omni/deploy/ming_tts.yaml \
  --enforce-eager
```

Run a streaming async-chunk case:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
  --model inclusionAI/Ming-omni-tts-0.5B \
  --case basic \
  --ref-audio /path/to/10002287-00000095.wav \
  --streaming \
  --deploy-config vllm_omni/deploy/ming_tts.yaml \
  --enforce-eager
```

The offline example includes 11 built-in cases: `style`, `ip`, `bgm`, `tta`,
`emotion`, `basic`, `dialect`, `zero_shot`, `podcast`, `speech_bgm`, and
`speech_sound`.

#### Online Command

Start the OpenAI-compatible speech server:

```bash
vllm-omni serve inclusionAI/Ming-omni-tts-0.5B \
  --deploy-config vllm_omni/deploy/ming_tts.yaml \
  --host 127.0.0.1 \
  --port 8091 \
  --enforce-eager \
  --omni \
  --stage-init-timeout 600 \
  --init-timeout 900 \
  --log-stats
```

Or use the bundled helper:

```bash
cd examples/online_serving/text_to_speech/ming_tts
./run_server.sh
```

#### Verification

Basic speech:

```bash
curl -X POST http://127.0.0.1:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "inclusionAI/Ming-omni-tts-0.5B",
    "input": "你好，这是 Ming 在线语音合成测试。",
    "response_format": "wav",
    "max_new_tokens": 200
  }' \
  --output ming_basic.wav
```

Style-conditioned speech:

```bash
curl -X POST http://127.0.0.1:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "inclusionAI/Ming-omni-tts-0.5B",
    "input": "我会一直在这里陪着你，直到你慢慢、慢慢地沉入那个最温柔的梦里……好吗？",
    "instructions": "{\"风格\":\"ASMR耳语，轻柔普通话，音量极低，语速极慢\"}",
    "response_format": "wav",
    "max_new_tokens": 200
  }' \
  --output ming_style.wav
```

Zero-shot cloning with reference audio and transcript:

```bash
curl -X POST http://127.0.0.1:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "inclusionAI/Ming-omni-tts-0.5B",
    "input": "我们的愿景是构建未来服务业的数字化基础设施。",
    "task_type": "Base",
    "ref_audio": "data:audio/wav;base64,<BASE64_WAV>",
    "ref_text": "在此奉劝大家别乱打美白针。",
    "response_format": "wav",
    "max_new_tokens": 200
  }' \
  --output ming_zero_shot.wav
```

Streaming PCM:

```bash
curl -X POST http://127.0.0.1:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "inclusionAI/Ming-omni-tts-0.5B",
    "input": "你好，这是 Ming 在线流式语音合成测试。",
    "instructions": "平静，普通话",
    "response_format": "pcm",
    "stream": true,
    "max_new_tokens": 200
  }' \
  --output ming_streaming.pcm
```

## Key Parameters

| Parameter | Scope | Description |
|---|---|---|
| `--deploy-config` | Offline / online | Use `vllm_omni/deploy/ming_tts.yaml` for the two-stage Ming pipeline |
| `--enforce-eager` | Offline / online | Recommended and used by the verified run |
| `--case` | Offline | Built-in case name from `cases.yaml` |
| `--streaming` | Offline | Uses `AsyncOmni` and async-chunk transfer |
| `voice` | Online | Selects a built-in IP voice such as `灵小甄` |
| `instructions` | Online | Free-form text or JSON-encoded Ming controls such as style, emotion, dialect, BGM, or environmental sound |
| `ref_audio` | Online | Reference audio, usually sent as a data URL for HTTP requests |
| `ref_text` | Online | Transcript paired with `ref_audio` for zero-shot cloning |
| `task_type` | Online | Use `Base` for reference-audio cloning requests |
| `response_format` | Online | `wav` for complete audio or `pcm` for streaming |
| `stream` | Online | Set `true` with `response_format="pcm"` for streaming output |
| `max_new_tokens` | Online | Upper bound for speech token generation |

## Verified Results

The following measurements came from the result summaries in
`/home/aja/Music/mingE2E27may` for commit
`4d923c708099939178e932ff153c63749b430fd1`. Each case used one warmup run and
one measured run on 1x A100 40GB. Memory peak was not available in the captured
stats.

### Offline

| Mode | Cases | E2E RTF | Elapsed range | TTFP |
|---|---:|---:|---:|---:|
| Blocking | 11 / 11 | 0.5011 - 0.6090, avg 0.5568 | 2.3541s - 15.0980s | N/A |
| Async chunk streaming | 11 / 11 | 0.4936 - 0.6079, avg 0.5468 | 2.2731s - 14.8571s | 2.2692s - 4.7519s, avg 4.0078s |

Offline blocking and async-chunk streaming both completed all 11 cases:
`style`, `ip`, `bgm`, `tta`, `emotion`, `basic`, `dialect`, `zero_shot`,
`podcast`, `speech_bgm`, and `speech_sound`.

### Online

Server startup was 110.01s. The `/v1/audio/speech` endpoint returned HTTP 200
for the warmup request, 9 WAV speech cases, and one streaming PCM request.

| Request group | Cases | E2E RTF / latency |
|---|---:|---:|
| WAV speech cases | 9 | RTF 0.5208 - 1.6646, avg 0.7622; elapsed 2.38s - 15.98s |
| Streaming PCM smoke test | 1 | elapsed 2.43s; TTFP 2.423s |

Online WAV cases verified: `style`, `ip`, `basic`, `emotion`, `dialect`,
`zero_shot`, `podcast`, `speech_bgm`, and `speech_sound`.

## Notes

- The deploy config sets `async_chunk: true`, `dtype: bfloat16`, and
  `trust_remote_code: false`.
- Stage 0 and Stage 1 both run on logical device `0` in the bundled config.
- The verified online route skips `bgm` and `tta` because `/v1/audio/speech`
  does not yet expose `prompt_mode=music` or `prompt_mode=tta`.
- Reference-audio fixtures used by the validation come from
  `inclusionAI/Ming-omni-tts/data/wavs`.
