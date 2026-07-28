# Run the MiniCPM-o 4.5 H100 regression trace

Run the command from the vLLM-Omni repository root inside the ROCm Docker container. The command uses the one GPU deployment and sends one request with the original H100 failure prompt. The corrected test explicitly requests text and audio with `use_tts_template=true` and `enable_thinking=false`, matching the original MiniCPM-o speech flow.

```bash
set -o pipefail

MINICPM_TRACE_DIR="$PWD/mi300x_trace_50_word_fix"
mkdir -p "$MINICPM_TRACE_DIR"

MINICPMO45_TRACE=1 \
MINICPMO45_E2E_DEPLOY_CONFIG=minicpmo_4_5.yaml \
TMPDIR="$MINICPM_TRACE_DIR" \
pytest -sv \
  'tests/e2e/online_serving/test_minicpmo_4_5_expansion.py::test_text_to_audio_long_output_001[default]' \
  --run-level full_model \
  2>&1 | tee "$MINICPM_TRACE_DIR/pytest.log"
```

The prompt is `Tell me a short story about a cat in about 50 words.` The request closes the thinking block before `<|tts_bos|>`, so the Talker receives only the generated answer.

The command writes the pytest log, temporary audio, and stage logs under `$PWD/mi300x_trace_50_word_fix`. Extract the trace records after the test finishes.

```bash
rg --no-filename '\[MiniCPMO45Trace\]' \
  "$MINICPM_TRACE_DIR"/omni_stage_*.log \
  > "$MINICPM_TRACE_DIR/trace.log"
```

Create an archive to share with the investigation.

```bash
tar -czf "$PWD/mi300x_trace_50_word_fix.tar.gz" \
  -C "$MINICPM_TRACE_DIR" .
```

Use the trace records to compare these boundaries for the same request ID:

```bash
rg '"event": "(thinker_to_talker|talker_terminal|talker_to_code2wav|code2wav_output|serving_audio)"' \
  "$MINICPM_TRACE_DIR/trace.log"
```

`thinker_to_talker` records the generated token IDs, stop reasons, boundary positions, thinking text, answer text, and handoff hashes. `talker_terminal` records whether the Talker stopped at EOS or at its audio token limit. `talker_to_code2wav` records the Talker scheduler state and codec hashes. The Code2Wav and serving records contain waveform counts and hashes.

Do not run another model request unless the trace is incomplete. The test sends one request.
