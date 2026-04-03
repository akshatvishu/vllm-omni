# Ming-omni-tts

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/ming_tts>.

This directory contains an offline Ming example that uses the in-repo Ming prompt builder directly. It covers zero-speaker style and IP generation, music generation, and reference-audio cloning flows.

## Quick Start

Run a zero-speaker style case:

```bash
python examples/offline_inference/ming_tts/end2end.py \
    --case style \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_tts.yaml
```

Run dialect cloning:

```bash
python examples/offline_inference/ming_tts/end2end.py \
    --case dialect \
    --ref-audio /path/to/yue_prompt.wav \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_tts.yaml
```

Run zero-shot cloning with a transcript:

```bash
python examples/offline_inference/ming_tts/end2end.py \
    --case zero_shot \
    --ref-audio /path/to/reference.wav \
    --ref-text "在此奉劝大家别乱打美白针。" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_tts.yaml
```

## Built-in Cases

- `style`: zero-speaker style-conditioned speech
- `ip`: zero-speaker IP voice generation
- `bgm`: music generation
- `basic`: reference-audio cloning with speed / pitch / volume control
- `dialect`: reference-audio cloning with dialect control
- `zero_shot`: reference-audio cloning with explicit transcript

## Streaming

Use async_chunk streaming with `AsyncOmni`:

```bash
python examples/offline_inference/ming_tts/end2end.py \
    --case style \
    --streaming \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_tts_async_chunk.yaml
```

## Example materials

??? abstract "README.md"
    ``````md
    --8<-- "examples/offline_inference/ming_tts/README.md"
    ``````
??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/ming_tts/end2end.py"
    ``````
