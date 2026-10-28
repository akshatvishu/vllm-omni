# All-layer functional AITER stage profile

Profile before selecting the next optimization. Main e6a7e63d plus the exact
functional AITER wrapper validated at d5f8de1e with 157/157 one-block checks.
Keep default V2 auto-functionalization enabled and ordinary regional compilation.
No P1 retained-prefix or direct-output optimization. Explicit FLASH_ATTN bound
to installed AITER, with SDPA calls forbidden. Wrapper audit instrumentation is
disabled; the wrapper's runtime validation remains part of the measured path.

Use a single gfx942 MI300X, BF16, batch one, TP1/SP1, both CFG branches, all 32
checkpoint blocks and the saved four-reference encoded context. Use the
production diffuse loop for the first three entries of the saved 40-step
schedule. Keep scheduler initialization at 40 steps. This covers prefill, first
decode including graph setup, and one steady decode. Exclude encoder, VAE,
HTTP and final-image quality. This is a new working configuration, not a speed
comparison against the failed native compiled baseline or historical SDPA runs.

Budget: one compilation warmup, two unprofiled stage measurements, then one
torch.profiler CPU/GPU trace with shapes and no stacks. Maximum subprocess time
900 seconds. Fresh request cache and captured graphs each run, as in main;
cleanup before each run is outside its measured interval. Stop after reporting
this experiment; no kernel sweep or full 40-step campaign follows automatically.

Acceptance:

- Pinned stack, wrapper hash, context hash, actual AITER binding and all weights.
- Six prediction calls, two captures with two warmups each, finite final latents.
- Repeated third-step latent bytes match the compilation warmup; no new compiled
  graphs during timed/profiled runs and no compiler graph breaks.
- Record event elapsed intervals and CPU submission/capture wall times. Nested
  graph setup is included in first-decode latency, not an additional additive
  stage. Events include device idle time between their markers.
- Correlate GPU work with HIP launches and enclosing stage annotations. GPU
  timestamps need not fall inside the CPU submission interval. Preserve unknown
  attribution. Separate kernel sums, device busy unions and elapsed intervals.
- Verify the trace shows AITER and no SDPA. Inspect expected 32-block/two-branch
  prefill call counts and map GEMMs to source only where justified.

Analyze raw traces on the remote host. Keep raw artifacts remote and write local
findings with provenance. Restore remote main, verify no GPU owners and release
the controller's lock. Only after this profile select a specific kernel for
hardware counters if the measured bottleneck needs that investigation.
