# Ming-omni-tts

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/ming_tts>.

This example shows how to serve Ming through the OpenAI-compatible `/v1/audio/speech` endpoint. The server builds Ming prompts directly with the in-repo prompt builder, so online requests support Ming-specific structured controls instead of the Qwen placeholder path.

## Launch the Server

```bash
vllm-omni serve inclusionAI/Ming-omni-tts-0.5B \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_tts_async_chunk.yaml \
    --omni \
    --port 8091 \
    --enforce-eager
```

Or:

```bash
cd examples/online_serving/ming_tts
./run_server.sh
```

## Example Requests

Style-conditioned speech:

```bash
python speech_client.py \
    --text "我会一直在这里陪着你。" \
    --instructions "轻柔的ASMR耳语，慢速，贴近麦克风"
```

Structured Ming control:

```bash
python speech_client.py \
    --text "我觉得社会企业同个人都有责任" \
    --instruction-json '{"方言":"广粤话"}'
```

IP voice generation:

```bash
python speech_client.py \
    --text "这款产品的名字，叫变态坑爹牛肉丸。" \
    --voice 灵小甄
```

Reference-audio cloning:

```bash
python speech_client.py \
    --task-type Base \
    --text "我们的愿景是构建未来服务业的数字化基础设施。" \
    --ref-audio /path/to/reference.wav \
    --ref-text "在此奉劝大家别乱打美白针。"
```

## Field Mapping

For Ming, the generic OpenAI request fields map to Ming controls like this:

- `input` -> target text
- `instructions` -> Ming instruction string, or a JSON string for the structured Ming control object
- `voice` -> Ming `IP`
- `language` -> Ming `方言`
- `ref_audio` + `ref_text` -> reference-audio cloning
- `speaker_embedding` -> 192-d Ming speaker embedding

## Voice Listing

- `/v1/audio/voices` lists uploaded voices for Ming.
- Built-in Ming IP labels can still be used as `voice`, but they are not enumerated by the API.

## Example materials

??? abstract "README.md"
    ``````md
    --8<-- "examples/online_serving/ming_tts/README.md"
    ``````
??? abstract "speech_client.py"
    ``````py
    --8<-- "examples/online_serving/ming_tts/speech_client.py"
    ``````
