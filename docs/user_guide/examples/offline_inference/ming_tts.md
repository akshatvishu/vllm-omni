# Ming-omni-tts

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_speech/ming_tts>.

This directory contains an offline Ming example that uses the in-repo Ming prompt builder directly. It covers the broader upstream dense 0.5B surface: style, IP, music-only generation, TTA, emotion, dialect, zero-shot clone, podcast, speech+bgm, and speech+sound.

## Quick Start

Run a zero-speaker style case:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case style \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

Run emotion-controlled speech:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case emotion \
    --ref-audio /path/to/emotion_prompt.wav \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

Run zero-shot cloning with a transcript:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case zero_shot \
    --ref-audio /path/to/reference.wav \
    --ref-text "在此奉劝大家别乱打美白针。" \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

Run podcast generation:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case podcast \
    --ref-audio-paths /path/to/CTS-CN-F2F-2019-11-11-423-012-A.wav /path/to/CTS-CN-F2F-2019-11-11-423-012-B.wav \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

Run text-to-audio event generation:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case tta \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

Run with stats and a manifest:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case style \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager \
    --enable-stats \
    --stats-log-file output_audio/ming_style_pipeline.log \
    --metadata-json output_audio/ming_style_manifest.json
```

## Built-in Cases

- `style`: zero-speaker style-conditioned speech
- `ip`: zero-speaker IP voice generation
- `bgm`: music generation
- `tta`: text-to-audio event generation with FlowLoss controls
- `emotion`: reference-audio speech with emotion control
- `basic`: reference-audio cloning with speed / pitch / volume control
- `dialect`: reference-audio cloning with dialect control
- `zero_shot`: reference-audio cloning with explicit transcript
- `podcast`: multi-reference dialogue generation with automatic speaker embedding extraction
- `speech_bgm`: speech with background music conditioning
- `speech_sound`: speech with environment sound conditioning

## Streaming

Use async_chunk streaming with `AsyncOmni`:

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case basic \
    --ref-audio /path/to/10002287-00000095.wav \
    --streaming \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

`--streaming` currently supports one prompt per process invocation. Use
blocking mode for `--num-prompts > 1`.

## Validation matrix

The example is intended to cover the dense TTS workflows used by the Ming
validation helper:

| Case | Blocking | Async chunk | Extra inputs |
|---|---:|---:|---|
| `style` | Yes | Optional smoke test | none |
| `ip` | Yes | Optional smoke test | none |
| `bgm` | Yes | Optional smoke test | none |
| `tta` | Yes | Optional smoke test | none |
| `emotion` | Yes | Yes | reference WAV |
| `basic` | Yes | Yes | reference WAV |
| `dialect` | Yes | Yes | reference WAV |
| `zero_shot` | Yes | Yes | reference WAV and transcript |
| `podcast` | Yes | Yes | two reference WAVs |
| `speech_bgm` | Yes | Yes | reference WAV |
| `speech_sound` | Yes | Yes | reference WAV |

The offline example also exposes vLLM-Omni runtime/reporting controls such as:

- `--num-prompts`
- `--enable-stats`
- `--stats-log-file`
- `--metadata-json`
- `--stage-init-timeout`
- `--init-timeout`
- `--batch-timeout`
- `--worker-backend`
- `--ray-address`

## Example materials

??? abstract "README.md"
    ``````md
    --8<-- "examples/offline_inference/text_to_speech/ming_tts/README.md"
    ``````
??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_speech/ming_tts/end2end.py"
    ``````
