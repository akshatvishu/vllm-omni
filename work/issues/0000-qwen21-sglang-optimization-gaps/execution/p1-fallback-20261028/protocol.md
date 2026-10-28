# P1 fallback diagnostic, 28 October 2026

The lifecycle probe failed a combined output byte and stride comparison without a native fallback control. Determine whether the difference exists before capture failure or appears specifically after native or P1 failure.

P1 model and manager sources remain byte identical to the lifecycle experiment. Use its checkpoint, four-reference input, 40-step schedule, first native CFG Euler update, BF16, TORCH_SDPA, TP1/SP1 and regional compilation. Reuse the existing compiler caches. Exercise only the unconditional decode branch that failed. Add 0.0625 to cached K and 0.125 to cached V in every layer as in the lifecycle test. These controlled prefixes do not represent a newly encoded prompt.

## Fixed sequence and budget

1. Load the actual 32-block transformer and perform native prefill for both CFG branches to reproduce the first Euler update.
2. With the changed prefix, run native graph twice and P1 graph twice. Save each output immediately and check repeatability and parity separately.
3. Install an empty manager, which returns the native fallback signal without fault injection. Run full transformer fallback twice with regional compilation still enabled.
4. Clear and re-register native's entry, inject a Python exception after its decode body completes inside capture, verify the marker and tensor release, then run full transformer fallback twice.
5. Perform the identical injection and fallback comparisons for P1. Compare each post-failure output to the saved pre-failure native fallback and compare native with P1 directly.
6. Verify immutable prefix bytes and explicitly tear down the probe's managers. Restore remote main and verify GPU idle.

One process, one input, one branch, one native and one P1 injected failure, maximum 600 seconds. No lifecycle ownership or eviction sequence, recovery sequence, timing, profiler, encoder, VAE, images or full requests. The first fault may affect subsequent process state; native before/after and P1 before/after comparisons can reveal a shared change, but cannot prove P1 isolation from a preceding fault if native already drifts.

## Evaluation

Save original shape, dtype, strides, offset and finite status independently. Save output tensors and content/file hashes remotely. Compare logical bytes with uint8 views, differing byte and element counts, maximum absolute error and RMSE for finite matching tensors. Keep mismatches as observations so all matched controls run. Protocol failures stop the run. A completed diagnostic is not a correctness pass or speed qualification.

Native graph versus fallback differences before injection establish a pre-existing execution-path difference for this setup. Native and P1 fallback parity with the pre-failure native fallback excludes an observed failure-induced delta in this input. A P1-only delta points to the candidate path. Equal logical bytes with different strides identifies a metadata-only mismatch. Do not infer the underlying library or arithmetic cause from an output difference alone.

Raw tensors remain remotely. Record findings, exact source and artifact hashes locally. Report before further recovery checks or performance experiments.
