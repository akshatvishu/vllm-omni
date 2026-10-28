# ROCm prefill output experiment

The user explicitly requested ROCm attention, excluded SDPA testing, and asked
to optimize prefill. This is one bounded exploratory experiment. The human's
objective is lower Qwen-Image-2.1 latency on their single MI300X. The selected
variable is prefill attention latency. Report its findings before proceeding.

Baseline: main e6a7e63d, Omni FLASH_ATTN resolved to installed AITER. Use only
that attention backend in both arms. Patch PyTorch SDPA to raise, so a hidden
fallback cannot silently contaminate the comparison. No old SDPA timing is a
matched baseline for this experiment.

Hypothesis: writing attention segments directly into a fresh final output can
remove the concatenation and segment-output allocations while preserving the
same installed AITER dispatch, mask, scaling and BF16 conversion.

Capture first-layer prefill Q/K/V and span metadata for both CFG branches from
the actual checkpoint and saved four-reference context. Construct only the
first transformer block and load its real weights plus shared input modules;
stop at its attention boundary. Preserve packed-QKV V strides and offsets.
These are actual model operands but not a full compiled-model reproduction.

Reference: current Omni FLASH_ATTN forward on the frozen operands. Candidate:
functional custom operator, newly allocated output per call, pinned AITER
private _flash_attn_forward with out slices. Scope: gfx942, BF16, batch one,
32 heads, head dimension 128, SP1 and full prefill query. No cached output state.

Before timing, require baseline A/A bytes, baseline/candidate bytes, unchanged
inputs, real/fake output metadata, and changed-input repeatability. Run eager
and torch.compile with Inductor graph capture disabled, matching the regional
compile policy where applicable. A component graph is not a whole block.
Check changed-input HIP replay separately with both implementations.

Fixed timing budget: three warmups, eight alternating pairs of ten invocations
per arm, plus two native A/A pairs. Include output allocation, all attention
segments and final concatenation in the measured callable. Record device-event
time and synchronized wall time. Require all exactness checks and a positive
median paired saving larger than observed native A/A spread to retain this
candidate for another stage. No rerunning until a favorable result.

After timing, collect one trace per compiled arm and summarize attention kernel
names/counts, concatenation, copies and duration. Confirm candidate removes
concatenation without replacing attention with SDPA. Compilation/setup time is
not a steady-state speed claim. Keep raw artifacts remote; save findings here.

Budget: one process, maximum 12 minutes including import and compilation.
No full denoising, HTTP, VAE, image-quality or request-speedup claim. Stop and
report if numerical, dispatch, compile or output-buffer contracts fail.
