# MiniCPM sliding recompute testing

This procedure tests a long English text request against the opt-in sliding recompute deployment after the runtime implementation exists.

## Proof rule

Do not mark the feature working from a unit test, a successful server start, or a non-empty audio file.

The test passes only when the logs show correct boundaries and the complete long request produces continuous, intelligible audio without duplicated, missing, or truncated segments.

## Source procedure

The launch and request shape are adapted from [`run_issue_5259_english_trace.md`](../issue_5259_backup/run_issue_5259_english_trace.md).

That reference contains commands but no expected output, so this file adds explicit evidence collection and pass conditions.

## Preconditions

Run the test inside the GPU container used for MiniCPM-o 4.5.

Use the checkout under test and verify that the imported package resolves to that checkout.

```bash
cd /scratch/workspace/vllm-omni
git rev-parse --short HEAD
python -c 'import vllm_omni; print(vllm_omni.__file__)'
```

Stop any old vLLM server before starting the test.

The command below uses the opt-in overlay `vllm_omni/deploy/minicpmo_4_5_sliding.yaml` relative to the repository root. The overlay inherits the base config and enables MiniCPM sliding recompute only on Stage 1.

## Start the server

Run the server in the first shell and preserve the complete log.

```bash
cd /scratch/workspace/vllm-omni
set -o pipefail
export VLLM_LOGGING_LEVEL=INFO

vllm-omni serve openbmb/MiniCPM-o-4_5 \
  --omni \
  --port 28889 \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/minicpmo_4_5_sliding.yaml \
  --interleave-mm-strings \
  2>&1 | tee minicpm-sliding-server.log
```

Wait for the server-ready message before sending the request.

Record the exact server command, model revision, deployment file, GPU type, container image, and commit in the history file.

## Send the long request

Run the client in a second shell with a prompt that requires at least 1,200 English words and a natural ending.

```bash
cd /scratch/workspace/vllm-omni
set -o pipefail

python examples/online_serving/minicpmo/openai_chat_completion_client_for_multimodal_generation.py \
  --model openbmb/MiniCPM-o-4_5 \
  --query-type text \
  --port 28889 \
  --prompt "Write an original English story of at least 1,200 words. Use clear spoken language, maintain a consistent setting and character names, do not repeat sentences, do not ask follow up questions, and end the story naturally." \
  2>&1 | tee minicpm-sliding-client.log
```

Save the returned audio without trimming it and record its duration, sample rate, channel count, and file size.

Transcribe the exact returned file with the repository utility:

```bash
python t.py <sliding-audio-file> --model large \
  2>&1 | tee minicpm-sliding-whisper.log
```

## Check the diagnostic logging

The new Talker diagnostics log codec sampling at condition steps 1, 25, 100, and 500, plus every recompute boundary. They include the prefill source, recompute epoch, hidden-state checksums, EOS probabilities before and after filtering, EOS filtering flags, sampled ID, and repetition-history tail.

Run these commands after the request completes:

```bash
cd /scratch/workspace/vllm-omni

rg "\[MiniCPM-o\]\[Stage1\]\[codec-sampling\]" minicpm-sliding-server.log
rg "\[MiniCPM-o\]\[Stage1\]\[(condition-boundary|sliding-recompute-schedule|sliding-recompute-prefill-input|sliding-recompute-prefill)\]" minicpm-sliding-server.log
rg "\[MiniCPM-o\]\[Stage1\]\[codec-sampling\].*condition_index=(13|14|15)" minicpm-sliding-server.log
if rg -n "Traceback|EngineDeadError|500 Internal Server Error|ValueError: MiniCPM-o" minicpm-sliding-server.log; then
  echo "runtime failure found"
  exit 1
fi
```

For every `sliding_recompute` sample, verify `prefill_source=sliding_recompute`, an increased `recompute_epoch`, a bounded `condition_step`, and a matching preceding `sliding-recompute-prefill` line. Compare `raw_eos_logit`, `pre_filter_eos_prob`, `post_filter_eos_prob`, `eos_filtered`, `eos_removed_by_warper`, `hidden_checksum`, and `sampled_id` around the first capped condition.

## Native-cache A/B run

Stop the sliding server and repeat the same client request with the synchronous native-cache diagnostic overlay. This keeps Stage 1 scheduling fixed while leaving `minicpmo_sliding_recompute` disabled.

```bash
cd /scratch/workspace/vllm-omni
set -o pipefail
export VLLM_LOGGING_LEVEL=INFO

vllm-omni serve openbmb/MiniCPM-o-4_5 \
  --omni \
  --port 28889 \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/minicpmo_4_5_native_diagnostic.yaml \
  --interleave-mm-strings \
  2>&1 | tee minicpm-native-server.log
```

In a second shell, run the identical request and save its output as `minicpm-native-client.log`:

```bash
cd /scratch/workspace/vllm-omni
set -o pipefail

python examples/online_serving/minicpmo/openai_chat_completion_client_for_multimodal_generation.py \
  --model openbmb/MiniCPM-o-4_5 \
  --query-type text \
  --port 28889 \
  --prompt "Write an original English story of at least 1,200 words. Use clear spoken language, maintain a consistent setting and character names, do not repeat sentences, do not ask follow up questions, and end the story naturally." \
  2>&1 | tee minicpm-native-client.log
```

Transcribe the returned native audio with `python t.py <native-audio-file> --model large`. A diagnostic comparison is evidence only; the feature remains unproven until the complete sliding audio is intelligible and has no repeated, missing, or truncated segment.

## Required log checks

Confirm that every request ID has a Thinker-to-Talker handoff before Talker prefill.

Confirm that Talker ordinary condition boundaries identify either `audio_eos` or `audio_limit_500`, and that condition indices increase without skips.

Confirm that every sliding recompute boundary reports the previous condition, previous audio-code count, current condition, exact replacement prompt length, and a reset of the computed-token offset to zero.

Confirm that the replacement prompt length stays within the configured Talker position limit and does not grow with the total request length.

Confirm that each Code2Wav codec sequence is monotonic within its cache epoch and that a final chunk is emitted only at the true request or turn boundary.

Confirm that the final request cleanup removes Talker and Code2Wav state for the request ID.

## Required negative checks

Fail the test if any condition boundary has no reason, condition index, or request ID.

Fail the test if any recompute boundary reuses only the 15-token repetition history instead of the full previous audio context.

Fail the test if Code2Wav receives a reordered chunk, a stale cache epoch, a duplicated terminal turn, or a terminal marker before the final request boundary.

Fail the test if the request stops at the old 4,096-token Talker budget before the text finishes, unless that limit was explicitly configured as the test budget.

Fail the test if concurrent requests show state or audio crossing between request IDs.

## Evidence bundle

Collect the server log, client log, deployment YAML, commit hash, package path, request text, generated audio, audio metadata, and a short boundary table.

Use one row per Talker condition with request ID, condition index, reason, generated codec count, replacement prompt length, reset offset, Code2Wav epoch, and chunk sequence.

## Pass criteria

Mark the feature passed only when all required log checks and negative checks pass and the long audio is complete and intelligible on manual listening.

If testing disproves a statement in [`history.md`](history.md), strike through the original statement and add the observed correction with the log or command that disproved it.

## Current status

**Implemented but unproven:** The opt-in runtime is implemented, but this document does not count as proof until the GPU E2E pass criteria above are met.
