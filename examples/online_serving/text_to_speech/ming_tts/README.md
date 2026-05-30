# Ming-omni-tts Online Serving

Serve the dense `inclusionAI/Ming-omni-tts-0.5B` two-stage TTS model through
the OpenAI-compatible `/v1/audio/speech` endpoint.

## Start Server

```bash
vllm-omni serve inclusionAI/Ming-omni-tts-0.5B \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --omni \
    --port 8091 \
    --enforce-eager
```

Or:

```bash
cd examples/online_serving/text_to_speech/ming_tts
./run_server.sh
```

## Send Requests

The Python client targets `http://localhost:8091/v1` with `api_key=EMPTY`; it
does not call OpenAI's hosted API.

```bash
python openai_speech_client.py \
    --text "你好，这是 Ming 在线语音合成测试。" \
    --max-new-tokens 200
```

Style or dialect controls can be plain text or Ming JSON:

```bash
python openai_speech_client.py \
    --text "我觉得社会企业同个人都有责任" \
    --instruction-json '{"方言":"广粤话"}' \
    --max-new-tokens 200
```

Reference-audio cloning:

```bash
python openai_speech_client.py \
    --task-type Base \
    --text "我们的愿景是构建未来服务业的数字化基础设施。" \
    --ref-audio /path/to/reference.wav \
    --ref-text "在此奉劝大家别乱打美白针。" \
    --max-new-tokens 200
```

Podcast-style multi-speaker prompt:

```bash
python openai_speech_client.py \
    --text "speaker_1:你可以说一下。 speaker_2:我也不知道。" \
    --ref-audio /path/to/speaker_1.wav \
    --ref-audio /path/to/speaker_2.wav \
    --ref-text "speaker_1:你好。 speaker_2:你好。"
```

Streaming PCM:

```bash
python openai_speech_client.py \
    --text "你好，这是流式输出测试。" \
    --stream \
    --output ming_output.pcm
```

`run_curl.sh` keeps small smoke checks:

```bash
./run_curl.sh basic
REF_AUDIO=/path/to/reference.wav REF_TEXT="在此奉劝大家别乱打美白针。" ./run_curl.sh zero_shot
./run_curl.sh stream
```

## Request Fields

| Field | Ming meaning |
|-------|--------------|
| `input` | target text |
| `instructions` | plain style text, or JSON object for structured Ming controls |
| `voice` | Ming IP voice label unless it resolves to an uploaded speaker |
| `language` | Ming `方言` control |
| `ref_audio` | prompt waveform; repeat/list values for podcast prompts |
| `ref_text` | transcript for zero-shot or podcast cloning |
| `speaker_embedding` | 192-d Ming speaker embedding |
| `max_new_tokens` | Ming `max_decode_steps` |

## Notes

- `ref_audio` accepts local paths through the client, remote URLs, `file://`,
  or `data:` URLs.
- Non-streaming responses return WAV bytes; streaming responses return PCM.
- Music-only `bgm` generation is offline-only until the API exposes Ming
  prompt-mode selection.
