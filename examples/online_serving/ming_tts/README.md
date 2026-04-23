# Ming-omni-tts

## Installation

Please refer to [README.md](../../../README.md)

## Model

| Model | Description |
|-------|-------------|
| `inclusionAI/Ming-omni-tts-0.5B` | Dense 0.5B Ming two-stage TTS model for speech generation with dialect, style, IP voice, and cloning controls |

## Launch the Server

```bash
vllm-omni serve inclusionAI/Ming-omni-tts-0.5B \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --omni \
    --port 8091 \
    --enforce-eager
```

Or use the convenience script:

```bash
cd examples/online_serving/ming_tts
./run_server.sh
```

The recommended online-serving path is eager async-chunk mode through
`/v1/audio/speech`. `run_server.sh` defaults to:

- model: `inclusionAI/Ming-omni-tts-0.5B`
- deploy config: `vllm_omni/deploy/ming_tts.yaml`
- auth: local testing only, no real OpenAI key required

## Send Requests

The canonical Ming online client is:

```bash
cd examples/online_serving/ming_tts
python openai_speech_client.py --text "你好，世界"
```

This talks to the local vLLM-Omni server at `http://localhost:8091/v1` and
uses `api_key=EMPTY`. It does not call OpenAI's cloud API.

### Basic TTS

```bash
python openai_speech_client.py \
    --text "你好，这是 Ming 在线语音合成测试。" \
    --max-new-tokens 200
```

### Style-conditioned speech without a reference clip

```bash
python openai_speech_client.py \
    --text "我会一直在这里陪着你。" \
    --instructions "轻柔的ASMR耳语，慢速，贴近麦克风" \
    --max-new-tokens 200
```

### Structured Ming control via JSON

```bash
python openai_speech_client.py \
    --text "我觉得社会企业同个人都有责任" \
    --instruction-json '{"方言":"广粤话"}' \
    --max-new-tokens 200
```

### IP voice generation

```bash
python openai_speech_client.py \
    --text "这款产品的名字，叫变态坑爹牛肉丸。" \
    --voice 灵小甄 \
    --max-new-tokens 200
```

### Reference-audio cloning

Ming has two reference-audio paths:

- prompt-waveform conditioning, where `ref_audio` steers the voice/style and
  `ref_text` is not required
- transcript cloning, where `ref_audio` and `ref_text` are paired

```bash
python openai_speech_client.py \
    --task-type Base \
    --text "我们的愿景是构建未来服务业的数字化基础设施。" \
    --ref-audio /path/to/reference.wav \
    --max-new-tokens 200
```

Pass `--ref-text` when the prompt case needs a transcript, such as zero-shot
voice cloning:

```bash
python openai_speech_client.py \
    --task-type Base \
    --text "我们的愿景是构建未来服务业的数字化基础设施。" \
    --ref-audio /path/to/reference.wav \
    --ref-text "在此奉劝大家别乱打美白针。" \
    --max-new-tokens 200
```

### Podcast-style multi-speaker prompt

```bash
python openai_speech_client.py \
    --text "speaker_1:你可以说一下。 speaker_2:我也不知道。" \
    --ref-audio /path/to/speaker_1.wav \
    --ref-audio /path/to/speaker_2.wav \
    --ref-text "在此奉劝大家别乱打美白针。"
```

### x-vector style cloning with a precomputed embedding

```bash
python openai_speech_client.py \
    --task-type Base \
    --text "你好，这是一段使用说话人向量的合成语音。" \
    --speaker-embedding /path/to/ming_speaker_embedding.json \
    --max-new-tokens 200
```

### Curl examples

`run_curl.sh` is intentionally small now. It keeps only three sanity checks:

```bash
./run_curl.sh basic
REF_AUDIO=/path/to/reference.wav REF_TEXT="在此奉劝大家别乱打美白针。" ./run_curl.sh zero_shot
./run_curl.sh stream
```

For the broader request cookbook, use direct `curl` payloads in this README.

Basic speech:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{
        "model": "inclusionAI/Ming-omni-tts-0.5B",
        "input": "你好，这是 Ming 在线语音合成测试。",
        "response_format": "wav"
    }' \
    --output ming_output.wav
```

Style-conditioned speech:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{
        "model": "inclusionAI/Ming-omni-tts-0.5B",
        "input": "我会一直在这里陪着你。",
        "instructions": "轻柔的ASMR耳语，慢速，贴近麦克风",
        "response_format": "wav"
    }' \
    --output ming_style.wav
```

IP voice generation:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{
        "model": "inclusionAI/Ming-omni-tts-0.5B",
        "input": "这款产品的名字，叫变态坑爹牛肉丸。",
        "voice": "灵小甄",
        "response_format": "wav"
    }' \
    --output ming_ip.wav
```

Dialect control with structured instructions:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{
        "model": "inclusionAI/Ming-omni-tts-0.5B",
        "input": "我觉得社会企业同个人都有责任",
        "instructions": "{\"方言\":\"广粤话\"}",
        "ref_audio": "data:audio/wav;base64,<BASE64_WAV>",
        "response_format": "wav"
    }' \
    --output ming_dialect.wav
```

Zero-shot cloning with transcript:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{
        "model": "inclusionAI/Ming-omni-tts-0.5B",
        "input": "我们的愿景是构建未来服务业的数字化基础设施。",
        "ref_audio": "data:audio/wav;base64,<BASE64_WAV>",
        "ref_text": "在此奉劝大家别乱打美白针。",
        "response_format": "wav"
    }' \
    --output ming_zero_shot.wav
```

Podcast-style multi-speaker prompt:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{
        "model": "inclusionAI/Ming-omni-tts-0.5B",
        "input": "speaker_1:你可以说一下。 speaker_2:我也不知道。",
        "ref_audio": [
            "data:audio/wav;base64,<BASE64_SPK1>",
            "data:audio/wav;base64,<BASE64_SPK2>"
        ],
        "ref_text": "speaker_1:你好。 speaker_2:你好。",
        "response_format": "wav"
    }' \
    --output ming_podcast.wav
```

Speaker-embedding cloning:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{
        "model": "inclusionAI/Ming-omni-tts-0.5B",
        "input": "你好，这是一段使用说话人向量的合成语音。",
        "speaker_embedding": [0.0, 0.0, 0.0],
        "response_format": "wav"
    }' \
    --output ming_embedding.wav
```

Streaming PCM response:

```bash
curl -N -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{
        "model": "inclusionAI/Ming-omni-tts-0.5B",
        "input": "你好，这是流式测试。",
        "stream": true,
        "response_format": "pcm"
    }' \
    --output ming_stream.pcm
```

## Request Types

Ming online serving supports these main request families through
`/v1/audio/speech`:

| Case | Online support | Required fields |
|------|----------------|-----------------|
| default TTS | Supported | `input`, `max_new_tokens=200` |
| `style` | Supported | `input`, `instructions`, `max_new_tokens=200` |
| `ip` | Supported | `input`, `voice`, `max_new_tokens=200` |
| `basic` helper | Supported | `input`, `max_new_tokens=200` |
| upstream `basic` case | Supported | `input`, `ref_audio`, structured speed / pitch / volume `instructions`, `max_new_tokens=200` |
| `emotion` | Supported | `input`, `ref_audio`, structured emotion `instructions`, `max_new_tokens=200` |
| `dialect` | Supported | `input`, `language` or structured `instructions`, `ref_audio`, `max_new_tokens=200` |
| `zero_shot` | Supported | `input`, `ref_audio`, `ref_text`, `max_new_tokens=200` |
| `podcast` | Supported | `input`, repeated/list `ref_audio`, `ref_text`, `max_new_tokens=200` |
| `speech_bgm` | Supported | `input`, `ref_audio`, structured `instructions` with `{"BGM": ...}`, `max_new_tokens=200` |
| `speech_sound` | Supported | `input`, `ref_audio`, structured `instructions` with `{"BGM": {"ENV": ...}}`, `max_new_tokens=200` |
| `bgm` | Not supported online | Requires a future `prompt_mode=music` API extension |

This matrix intentionally mirrors the local online validation flow. The
music-only `bgm` case remains offline-only because `/v1/audio/speech` always
uses Ming's speech prompt path today.

## Output

- Non-streaming requests return full audio bytes, usually written to `.wav`
- WAV outputs are expected to be readable at 44.1kHz
- Streaming requests return progressive PCM bytes; wrap or convert them to WAV
  before browser playback
- The default Python client outputs:
  - `ming_output.wav` for non-streaming
  - `ming_output.pcm` for streaming

## Validated Outputs

Validation on an L4 GPU passed the online async_chunk `/v1/audio/speech` flow
for every speech-mode case in the local validation script:

| Case | Output | Size bytes | Sample rate | Frames |
|------|--------|-----------:|------------:|-------:|
| `style` | WAV | 790316 | 44100 | 395136 |
| `ip` | WAV | 366956 | 44100 | 183456 |
| `basic` | WAV | 536300 | 44100 | 268128 |
| `emotion` | WAV | 649196 | 44100 | 324576 |
| `dialect` | WAV | 395180 | 44100 | 197568 |
| `zero_shot` | WAV | 931436 | 44100 | 465696 |
| `podcast` | WAV | 846764 | 44100 | 423360 |
| `speech_bgm` | WAV | 677420 | 44100 | 338688 |
| `speech_sound` | WAV | 649196 | 44100 | 324576 |
| `streaming` | PCM | 338688 | N/A | N/A |

`bgm` is intentionally not included in the online pass list. It is a
music-prompt workflow, while `/v1/audio/speech` currently routes Ming through
the speech prompt path.

## Performance

Benchmark via `/v1/audio/speech`, `inclusionAI/Ming-omni-tts-0.5B`,
10 prompts, concurrency 1, eager mode:

| Config | Mean TTFP | Mean E2E | Mean RTF |
|--------|----------:|---------:|---------:|
| Sequential eager | 3354.83ms | 3357.01ms | 0.561 |
| Async chunk eager | 3450.28ms | 3452.35ms | 0.577 |

## Audio Inputs

- `ref_audio` accepts:
  - a local file path
  - a remote `http://` or `https://` URL
  - a `data:` URL
  - repeated values for podcast-style multi-speaker prompts
- `openai_speech_client.py` converts local reference audio files into a base64
  `data:` URL before sending them to the server
- `speaker_embedding` must be a JSON file containing exactly 192 numeric values
- Ming prompt-waveform cases can use `ref_audio` without `ref_text`
- Zero-shot and podcast-style transcript cloning should include `ref_text`

## API Field Mapping

The OpenAI-compatible `/v1/audio/speech` endpoint stays generic. Ming-specific controls are mapped like this:

- `input` -> target text
- `instructions` -> Ming instruction string, or a JSON string that becomes the structured Ming control object
- `voice` -> Ming `IP` field when using built-in character voices
- `language` -> Ming `方言` field
- `ref_audio` -> Ming `prompt_waveform`
- `ref_text` -> Ming `prompt_text`
- `speaker_embedding` -> 192-d Ming speaker embedding
- `max_new_tokens` -> Ming `max_decode_steps`

## Voice Listing

- `/v1/audio/voices` reflects uploaded voices for Ming.
- Built-in Ming IP labels like `灵小甄` are passed through as `voice` values, but they are not enumerated by the API.

## Streaming

Use `stream=true` to get progressive PCM output:

```bash
python openai_speech_client.py \
    --text "你好，这是流式输出测试。" \
    --instructions "平静，普通话" \
    --stream \
    --output ming_output.pcm
```

## Not Supported Online Yet

`bgm` music-prompt generation is not exposed through `/v1/audio/speech` today.
It needs a future `prompt_mode=music` API extension so the server can select
Ming's music system prompt instead of the speech system prompt.

## Troubleshooting

### No real OpenAI key

The example targets a local vLLM-Omni server. `api_key=EMPTY` is expected and
is sufficient for local testing.

### `--ref-audio` fails

- Confirm the local file exists
- If using zero-shot or podcast transcript cloning, also provide `--ref-text`
- If passing a URL, make sure the server can fetch it

### `--speaker-embedding` fails

- Make sure the JSON file contains exactly 192 numeric values
- Do not wrap the list in another object

### Connection refused

- Check that the server is running on `localhost:8091`
- Confirm the stage config path is correct

### No audio or wrong output file

- Use non-streaming for `.wav`
- Use `--stream` for `.pcm`

### `bgm` is missing online

Use the offline example for music-only `bgm`. Online support needs an explicit
Ming prompt-mode API extension so the server can select the music prompt
instead of the speech prompt.
