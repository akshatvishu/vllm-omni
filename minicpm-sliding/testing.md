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

The sliding deployment overlay must exist before running the command below; replace `<sliding-deploy-config>` with its verified path.

## Start the server

Run the server in the first shell and preserve the complete log.

```bash
cd /scratch/workspace/vllm-omni
set -o pipefail
export VLLM_LOGGING_LEVEL=DEBUG

vllm-omni serve openbmb/MiniCPM-o-4_5 \
  --omni \
  --port 28889 \
  --trust-remote-code \
  --deploy-config <sliding-deploy-config> \
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

**Unproven:** The sliding recompute runtime and its E2E behavior are not proven by this document.
