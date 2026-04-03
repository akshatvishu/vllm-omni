# Ming-omni-tts

## Model

| Model | Description |
|-------|-------------|
| `inclusionAI/Ming-omni-tts-0.5B` | Dense 0.5B Ming two-stage TTS model for speech generation with dialect, style, IP voice, and cloning controls |

## Launch the Server

```bash
vllm-omni serve inclusionAI/Ming-omni-tts-0.5B \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_tts_async_chunk.yaml \
    --omni \
    --port 8091 \
    --enforce-eager
```

Or use the convenience script:

```bash
cd examples/online_serving/ming_tts
./run_server.sh
```

## Send Speech Requests

### Style-conditioned speech without a reference clip

```bash
python speech_client.py \
    --text "我会一直在这里陪着你。" \
    --instructions "轻柔的ASMR耳语，慢速，贴近麦克风"
```

### Structured Ming control via JSON

```bash
python speech_client.py \
    --text "我觉得社会企业同个人都有责任" \
    --instruction-json '{"方言":"广粤话"}'
```

### IP voice generation

```bash
python speech_client.py \
    --text "这款产品的名字，叫变态坑爹牛肉丸。" \
    --voice 灵小甄
```

### Reference-audio cloning

```bash
python speech_client.py \
    --task-type Base \
    --text "我们的愿景是构建未来服务业的数字化基础设施。" \
    --ref-audio /path/to/reference.wav \
    --ref-text "在此奉劝大家别乱打美白针。"
```

### x-vector style cloning with a precomputed embedding

```bash
python speech_client.py \
    --task-type Base \
    --text "你好，这是一段使用说话人向量的合成语音。" \
    --speaker-embedding /path/to/ming_speaker_embedding.json
```

## API Field Mapping

The OpenAI-compatible `/v1/audio/speech` endpoint stays generic. Ming-specific controls are mapped like this:

- `input` -> target text
- `instructions` -> Ming instruction string, or a JSON string that becomes the structured Ming control object
- `voice` -> Ming `IP` field when using built-in character voices
- `language` -> Ming `方言` field
- `ref_audio` + `ref_text` -> reference-audio cloning inputs
- `speaker_embedding` -> 192-d Ming speaker embedding

## Voice Listing

- `/v1/audio/voices` reflects uploaded voices for Ming.
- Built-in Ming IP labels like `灵小甄` are passed through as `voice` values, but they are not enumerated by the API.

## Streaming

Use `stream=true` to get progressive PCM output:

```bash
python speech_client.py \
    --text "你好，这是流式输出测试。" \
    --instructions "平静，普通话" \
    --stream \
    --output ming_output.pcm
```
