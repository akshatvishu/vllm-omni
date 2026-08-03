# MiniCPM sliding recompute history

This file records the evidence, changes, corrections, and open risks for MiniCPM-o 4.5 sliding recompute in vLLM-Omni.

## History rules

1. Every entry is marked `Observed`, `Proposed`, `Tested`, or `Unproven`.
2. Do not delete an entry when testing disproves it.
3. Strike through the disproved text with `~~...~~`, then add the correction and its evidence below it.
4. A unit test, log trace, or code inspection is not an end-to-end proof; only a recorded E2E pass proves the serving behavior.

## 2026-08-03 baseline

### Pipeline boundary map

| Stage | Current boundary | Evidence status |
| --- | --- | --- |
| 0 Thinker | Generates text and hidden states, then `llm2tts` extracts the TTS region for Stage 1. | Observed in [`pipeline.py`](../vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py) and [`llm2tts`](../vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py). |
| 1 Talker prefill | Builds text plus projected Thinker hidden-state embeddings and starts one condition chunk. | Observed in [`preprocess`](../vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py). |
| 1 Talker decode | Samples codec IDs until audio EOS or 500 codec steps for ordinary streaming, or 26 codec steps for native duplex. | Observed in [`make_omni_output`](../vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py). |
| 1 to 2 bridge | Accumulates codec deltas into 25-frame windows and adds three frames of left context. | Observed in [`tts2code2wav_async_chunk`](../vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py). |
| 2 Code2Wav | Validates request, epoch, sequence, prompt, and codec shape before decoding waveform. | Observed in [`MiniCPMO45Code2Wav`](../vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_code2wav.py). |

### Current cache behavior

**Observed:** Stage 1 uses a native vLLM `LlamaModel`, and ordinary condition transitions are injected by `preprocess` without resetting the vLLM request session.

**Observed:** The current scheduler replacement path resets prompt bookkeeping and model intermediate data, so it cannot be reused for internal recomputation without a state-preservation option.

**Observed:** The current 15-token Talker history is for repetition penalty and is not sufficient as the full previous audio context required by MiniCPM recomputation.

### Reference behavior

**Observed:** The reference `sliding_recompute` path clears `past_key_values`, rebuilds the previous condition plus previous generated audio embeddings plus the current condition, and restarts positions at zero.

**Observed:** The reference default uses one recomputed prior chunk and a two-chunk window, while the current vLLM-Omni Talker has no equivalent context mode yet.

Evidence: [`utils.py`](../issue_5259_backup/MiniCPM-o-4_5/utils.py) and [`modeling_minicpmo.py`](../issue_5259_backup/MiniCPM-o-4_5/modeling_minicpmo.py).

### Architecture decision

**Proposed:** Keep native vLLM KV caching for the default Talker path and add an internal model-to-scheduler prompt-replacement control for the opt-in sliding recompute path.

**Proposed:** The Talker owns request-keyed recompute state, while the scheduler owns prompt replacement and KV lifetime; no model code will mutate vLLM KV blocks directly.

**Proposed:** Do not move Thinker-to-Talker conditions onto a new async connector for this feature because only the Talker knows the exact number of previous audio codes needed to size the recompute prompt.

**Unproven:** The control event, scheduler requeue, Code2Wav segment handling, and long-request output budget still require implementation and E2E validation.

### Logging instrumentation

**Changed:** Added boundary logs for Thinker-to-Talker handoff, Talker prefill, ordinary condition EOS or 500-step termination, native duplex chunk termination, Talker cleanup, codec-window emission, and Code2Wav input validation.

**Changed:** The logs include request ID, stage boundary, condition index, condition length, prompt length, computed-token offset, codec count, cache epoch, chunk sequence, segment flags, and terminal flags where available.

**Observed:** The current ordinary condition log reports `kv_action=append_native_kv`; no sliding recompute reset event exists until the runtime control path is implemented.

**Unproven:** The current async bridge must be tested to verify that a condition boundary cannot be mistaken for a final Code2Wav chunk.

**Unproven:** Logging shows the control-flow evidence but does not prove audio correctness, cache correctness, or long-form continuity.

### Validation after instrumentation

**Tested:** The focused MiniCPM unit suite passed with 84 tests passed and 1 skipped after the logging changes.

Command: `.venv/bin/pytest -q tests/model_executor/models/minicpmo_4_5/test_talker_batching.py tests/model_executor/models/minicpmo_4_5/test_code2wav_batching.py tests/model_executor/models/minicpmo_4_5/test_llm2tts.py tests/model_executor/stage_input_processors/test_minicpmo_4_5_async_chunk.py`.

**Tested:** Ruff checks and formatting passed for all three modified Python files.

**Unproven:** These checks do not prove sliding recompute, long-form audio continuity, or an end-to-end server request.

## Future entry template

### YYYY-MM-DD short title

**Observed/Proposed/Tested/Unproven:** State one change or result in no more than two sentences.

Evidence: Link the exact file, command, log excerpt, or test result.

## E2E proof status

**Unproven:** No sliding recompute E2E request has passed yet.

The first accepted proof must include the server log, client log, commit, deployment configuration, request text, audio output, and a boundary-by-boundary result.
