# Ming-omni-tts

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/ming_tts>.

This example shows how to serve Ming through the OpenAI-compatible `/v1/audio/speech` endpoint. The server builds Ming prompts directly with the in-repo prompt builder, so online requests support Ming-specific structured controls instead of the Qwen placeholder path.

## Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

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

The canonical Ming online client is `openai_speech_client.py`. It targets the
local vLLM-Omni server, not OpenAI's cloud API, so `api_key=EMPTY` is enough
for local testing.

## Example Requests

Basic TTS:

```bash
python openai_speech_client.py \
    --text "你好，这是 Ming 在线语音合成测试。"
```

Style-conditioned speech:

```bash
python openai_speech_client.py \
    --text "我会一直在这里陪着你。" \
    --instructions "轻柔的ASMR耳语，慢速，贴近麦克风"
```

Structured Ming control:

```bash
python openai_speech_client.py \
    --text "我觉得社会企业同个人都有责任" \
    --instruction-json '{"方言":"广粤话"}'
```

IP voice generation:

```bash
python openai_speech_client.py \
    --text "这款产品的名字，叫变态坑爹牛肉丸。" \
    --voice 灵小甄
```

Reference-audio cloning:

Use `ref_audio` by itself for Ming prompt-waveform conditioning. Add
`ref_text` when the request is transcript cloning, such as zero-shot or
podcast-style prompts.

```bash
python openai_speech_client.py \
    --task-type Base \
    --text "我们的愿景是构建未来服务业的数字化基础设施。" \
    --ref-audio /path/to/reference.wav \
    --ref-text "在此奉劝大家别乱打美白针。"
```

Speaker-embedding cloning:

```bash
python openai_speech_client.py \
    --task-type Base \
    --text "你好，这是一段使用说话人向量的合成语音。" \
    --speaker-embedding /path/to/ming_speaker_embedding.json
```

Streaming PCM:

```bash
python openai_speech_client.py \
    --text "你好，这是流式输出测试。" \
    --instructions "平静，普通话" \
    --stream \
    --output ming_output.pcm
```

## Curl Helper

Use the bundled helper for common request types:

```bash
./run_curl.sh basic
./run_curl.sh style
./run_curl.sh ip
REF_AUDIO=/path/to/emotion_prompt.wav ./run_curl.sh emotion
REF_AUDIO=/path/to/yue_prompt.wav ./run_curl.sh dialect
REF_AUDIO=/path/to/reference.wav REF_TEXT="在此奉劝大家别乱打美白针。" ./run_curl.sh zero_shot
REF_AUDIO=/path/to/speaker_1.wav REF_AUDIO_2=/path/to/speaker_2.wav REF_TEXT="speaker_1:你好。 speaker_2:你好。" ./run_curl.sh podcast
REF_AUDIO=/path/to/00000309-00000300.wav ./run_curl.sh speech_bgm
REF_AUDIO=/path/to/00000309-00000300.wav ./run_curl.sh speech_sound
REF_AUDIO=/path/to/reference.wav REF_TEXT="在此奉劝大家别乱打美白针。" ./run_curl.sh clone_ref_audio
SPEAKER_EMBEDDING=/path/to/ming_speaker_embedding.json ./run_curl.sh clone_embedding
./run_curl.sh stream
```

## Audio Inputs

- `ref_audio` accepts a local path, remote URL, or `data:` URL
- The Python client converts local files into a base64 `data:` URL
- `speaker_embedding` must be a JSON file with exactly 192 numeric values
- Ming prompt-waveform cases can use `ref_audio` without `ref_text`
- Zero-shot and podcast-style transcript cloning should include `ref_text`

The bundled `run_curl.sh basic` mode is plain/default TTS and does not require
`REF_AUDIO`. The upstream cookbook-style `basic` case uses `ref_audio` plus
structured speed / pitch / volume instructions.

## Field Mapping

For Ming, the generic OpenAI request fields map to Ming controls like this:

- `input` -> target text
- `instructions` -> Ming instruction string, or a JSON string for the structured Ming control object
- `voice` -> Ming `IP`
- `language` -> Ming `方言`
- `ref_audio` -> Ming prompt waveform
- `ref_text` -> optional transcript for zero-shot and podcast-style cloning
- `speaker_embedding` -> 192-d Ming speaker embedding

## Voice Listing

- `/v1/audio/voices` lists uploaded voices for Ming.
- Built-in Ming IP labels can still be used as `voice`, but they are not enumerated by the API.

## Example materials

??? abstract "README.md"
    ``````md
    --8<-- "examples/online_serving/ming_tts/README.md"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/ming_tts/run_server.sh"
    ``````
??? abstract "openai_speech_client.py"
    ``````py
    --8<-- "examples/online_serving/ming_tts/openai_speech_client.py"
    ``````
??? abstract "run_curl.sh"
    ``````sh
    --8<-- "examples/online_serving/ming_tts/run_curl.sh"
    ``````
