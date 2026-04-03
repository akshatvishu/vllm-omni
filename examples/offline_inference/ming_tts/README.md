# Ming-omni-tts Offline Inference

`end2end.py` runs Ming-omni-tts end-to-end with vLLM Omni. It uses the in-repo Ming prompt builder directly, so the example request shape matches the actual integration instead of a simplified placeholder path.

## Supported Demo Cases

- `style`: zero-speaker style-conditioned speech
- `ip`: zero-speaker IP voice generation
- `bgm`: music generation
- `basic`: reference-audio cloning with speed / pitch / volume controls
- `dialect`: reference-audio cloning with dialect control
- `zero_shot`: reference-audio cloning with explicit transcript

## Quick Start

Run the zero-speaker style example:

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

Run zero-shot cloning with a reference transcript:

```bash
python examples/offline_inference/ming_tts/end2end.py \
    --case zero_shot \
    --ref-audio /path/to/reference.wav \
    --ref-text "在此奉劝大家别乱打美白针。" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_tts.yaml
```

Use async_chunk streaming:

```bash
python examples/offline_inference/ming_tts/end2end.py \
    --case style \
    --streaming \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_tts_async_chunk.yaml
```

## Key Arguments

| Argument | Description |
|---|---|
| `--model` | Hugging Face repo or local Ming checkpoint path |
| `--stage-configs-path` | Stage config YAML. Use `ming_tts.yaml` for blocking generation or `ming_tts_async_chunk.yaml` for streaming |
| `--case` | Built-in demo case: `style`, `ip`, `bgm`, `basic`, `dialect`, `zero_shot` |
| `--ref-audio` | Reference wav path for cloning cases |
| `--ref-text` | Reference transcript for zero-shot cloning |
| `--instructions` | Free-form Ming instruction string |
| `--instruction-json` | Structured Ming instruction JSON |
| `--speaker-embedding` | JSON file containing a 192-d speaker embedding |
| `--max-decode-steps` | Override `ming_max_decode_steps` |
| `--streaming` | Use `AsyncOmni` and async_chunk transport |

## Notes

- Ming cloning cases need reference audio. `zero_shot` also needs `--ref-text`.
- Zero-speaker cases use the integration’s `use_zero_spk_emb` path and do not require a reference clip.
- The script writes mono 44.1 kHz WAV output.
