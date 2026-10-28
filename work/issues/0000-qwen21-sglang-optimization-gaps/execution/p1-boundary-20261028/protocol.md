# Native graph/fallback boundary diagnostic, 28 October 2026

Find the first differing native compiled-block boundary without changing P1 or adding hooks inside compiled arithmetic. If all common inputs to that block match, test whether changing only `prepared_modulation` changes its output to the other path's output. Test `_pointwise_validated` separately.

Use the same actual checkpoint, saved context, first native CFG Euler update, controlled unconditional K/V additions, regional compilation, SDPA, BF16 and single MI300X as the preceding fallback diagnostic. Verify the saved hidden-input hash, all 64 prefix hashes and both uninstrumented output hashes before attribution. Keep P1 model/manager source byte identical, but execute only native entries with retained buffers disabled. No fault injection or P1 replay is needed again.

## Sequence

1. Reproduce native graph and ordinary fallback outputs twice. Reuse the existing compiler cache and compare against the previous saved hashes.
2. Wrap each already compiled block's callable externally. Copy block inputs/outputs to fixed buffers allocated before capture. Copy shared modulation, RoPE and token masks at block zero. Never hook child operations inside the compiled block.
3. Recapture native graph once with those external copies and run observed fallback once. Require final outputs to remain byte exact against their uninstrumented references. Otherwise stop attribution and report instrumentation sensitivity.
4. Compare block boundaries in execution order, stopping at the first difference. Save that block's inputs, prefix K/V, output and prepared modulation. Check common operand bytes and metadata separately.
5. Only if common operands match, replay the unchanged compiled block in each original inference context. Require each output to match its captured reference before performing argument swaps.
6. Swap only prepared modulation in each direction, then change only the validation flag in each direction. Compare each output to both references. Save loaded Inductor generated modules and compiler counters without modifying compiler configuration.
7. Check prefix hashes and restore remote main. Report before another experiment.

One process, one unconditional four-reference input, maximum 600 seconds. No request timing, VAE, encoder, image or 40-step run. All comparison failures remain recorded; protocol/reproduction failures stop the diagnostic. No repeated tuning until a result passes.

## Interpretation

Exact common inputs with differing block outputs localize the difference inside that compiled block. Reciprocal output changes from prepared-modulation swaps implicate the Omni preparation/compilation boundary at this input. A flag-only effect would identify a separate control. Generated code can establish which casts and arithmetic were emitted, but a module merely present in the compiler cache is not proof that a particular kernel executed. Do not assign a library bug or exact instruction-level cause without that evidence. Metadata differences, inference context and any compiler fallback must remain explicit.

Raw tensors and generated code stay remotely. Record local findings and artifact identities. This diagnostic does not qualify P1 lifecycle recovery, image quality or speed.
