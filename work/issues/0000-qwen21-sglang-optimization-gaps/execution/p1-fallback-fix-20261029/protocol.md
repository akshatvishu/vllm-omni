# Compiled fallback repair and recovery test, 29 October 2026

The preceding diagnostic isolated graph/fallback differences to the placement of modulation preparation. Repair only the ordinary forward preparation condition. During CUDA/ROCm decode, when all blocks expose the regional compiler's `_torchdynamo_orig_callable` marker, leave prepared modulation unset so each compiled block uses the graph body's arithmetic. Keep prefill, uncompiled decode, NPU preparation, the graph body and P1 retention code unchanged. No global compiler precision settings change.

## Fixed test sequence

1. Run the existing shared-modulation and decode-graph CPU tests in the container before model work. Load the actual 32-block checkpoint and reproduce the saved four-reference context, native CFG prefill and first Euler update.
2. Apply the same controlled +0.0625 K and +0.125 V additions in every layer and both CFG branches. Verify the saved unconditional prefix/input hashes.
3. Compare complete native graph, native A/A, P1 graph and repaired ordinary fallback for both CFG branches. Require byte, shape, dtype and stride equality, finite output, and preservation of the saved unconditional graph hash. Save outputs and report mismatch counts before stopping on a failure.
4. Observe the arguments to all 32 compiled blocks externally for one fallback in each branch. Preserve the compiler marker with `functools.wraps`. Require prepared modulation to be absent and final output to match the uninstrumented graph reference.
5. In native and P1 separately, replace only the probe's unconditional entry to force fresh capture. Keep the conditional graph alive. Inject a Python exception after the third actual decode body, inside capture. Require fallback signal, bounded failed marker, no retry and release of observed failed-entry tensors. Run actual full forward fallback twice and compare with the graph baseline.
6. After each failure, replay both managers' conditional survivors. Require exact output, identical retained addresses and unchanged capture count. Verify request prefix hashes.
7. Release the old request owner, call the production idle cleanup API and require failure markers, survivor entries, owner records and observed tensor objects to be released. Register a new owner with the same geometry, cloned first K and +0.25 first-layer V. Require fresh native/P1 captures, changed outputs, exact native/P1/repaired-fallback parity, exact repeats and unchanged new prefix hashes in both CFG branches.
8. Release the new request, call idle cleanup and verify final entry/tensor release. Restore remote main and confirm GPU idle.

One process and one four-reference geometry on the authorized MI300X, with the existing stack and compiler caches. Worker timeout is 600 seconds, including at most 120 seconds for existing CPU tests. No 40-step denoising, encoder, VAE, image, profiling, timing or peak-memory test. Do not repeat the previously passed live-owner conflict or LRU-capacity sequence.

## Evaluation and limits

All parity and lifecycle checks must pass. Preserve failures and report them before another experiment. Passing verifies this saved input, both CFG branches and the stated failure/recovery sequence. The injected exception is a Python failure after a completed captured body, not a device fault or OOM. Tensor release is checked with weak references and bounded manager state, not allocator-reserved or process-peak memory.

Idle cleanup is explicitly invoked through the existing API after request release. This does not claim automatic exception cleanup or server cancellation/sleep behavior. No NVIDIA, NPU, offload, SP, mixed compilation or alternate attention backend qualification is implied. The repair's source condition covers CUDA/ROCm direct regional compilation; this experiment validates only the stated MI300X configuration. Save raw artifacts remotely and write findings locally. Report before any speed experiment.
