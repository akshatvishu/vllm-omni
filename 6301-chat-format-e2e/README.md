# Issue 6301 chat format experiments

The bundle records the wire response and the OpenAI Python SDK response for the chat completions cases needed by issue 6301.

Run it from the repository root inside the vLLM-Omni Docker container:

```bash
bash 6301-chat-format-e2e/run_all.sh
```

The default Qwen3-Omni deployment uses two visible GPUs. Select different physical GPUs without editing the repository paths:

```bash
GPU_IDS=2,3 BATCH_SIZE=2 bash 6301-chat-format-e2e/run_all.sh
```

Each run is written to `6301-chat-format-e2e/runs/<timestamp>/`. It contains the complete runner and server logs, environment and Git snapshots, request bodies, raw HTTP responses, typed SDK responses, and `summary.json`.

The runner continues after a failed cell and exits with status 1 after it saves the full summary. A Pydantic serializer type mismatch counts as a typed SDK failure because the response does not match the SDK field type.

The matrix is:

| Model | Client | Stream | Concurrent batch | Output under test |
|---|---|---:|---:|---|
| Qwen3-Omni | Raw HTTP | No | No | Text and audio from the three stage pipeline |
| Qwen3-Omni | OpenAI SDK | No | No | Text and audio from the three stage pipeline |
| Qwen3-Omni | Raw HTTP | Yes | No | Text and audio stream chunks |
| Qwen3-Omni | OpenAI SDK | Yes | No | Text and audio stream chunks |
| Qwen3-Omni | Raw HTTP | No and yes | Yes | Concurrent response isolation |
| Qwen3-Omni | OpenAI SDK | No and yes | Yes | Concurrent response isolation |
| Z-Image | Raw HTTP | No | No | Two image content parts |
| Z-Image | OpenAI SDK | No | No | Strict typed parsing of two image content parts |

Useful overrides are `QWEN_MODEL`, `ZIMAGE_MODEL`, `QWEN_DEPLOY_CONFIG`, `PORT`, `BATCH_SIZE`, `SERVER_START_TIMEOUT`, `REQUEST_TIMEOUT`, `RUN_QWEN=0`, and `RUN_ZIMAGE=0`.
