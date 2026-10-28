# Functional AITER boundary, one-block validation

Hypothesis: accurately declaring AITER forward attention functional avoids the
mutation-driven copies and V2 decomposition failure while preserving arithmetic.
The wrapper calls the same public AITER function without out= and returns its
fresh output. Main's piecewise loop and concatenation remain unchanged.

Use main e6a7e63, one gfx942 MI300X, pinned installed Torch/AITER, BF16, batch one,
TP1/SP1 and the real first-block checkpoint weights. Capture block inputs from
the saved four-reference context for both CFG branches. Validate prefill and
one decode geometry. Decode uses the saved initial latents at timestep 1; it is
a representative integration test, not a completed denoising trajectory.

Reference: compile the surrounding identical block with original whole attention
executed inside a separate functional opaque operator. This preserves compiled
producer arithmetic and avoids executing the known unsafe native compile path.
Also compare original eager block with eager candidate. Do not require eager
block arithmetic to equal compiled arithmetic or label this a working original
compiled B0. Compare full block outputs and populated prefix caches bytewise.

Acceptance gates:

1. Verified source, saved-context hash, device, actual AITER binding and all used
   weights loaded. Default V2 stays enabled; no V1 compatibility override.
2. Eager original/candidate exactness. Compiled reference A/A, candidate repeat
   and reference/candidate exactness for both CFG branches and both phases.
3. Actual compiled Q/K/V segment operands match the compiled reference operands
   bytewise. Q/K/V unchanged by AITER calls. Output is fresh and fake/real shape,
   dtype, device and strides match for actual CK, ASM and dense calls.
4. Hidden states, modulation, frequencies and decode cache inputs unchanged.
   Changed-hidden-input compiled comparisons and changed-input HIP graph replay
   exact. Graphs use warmed compiled callables, without audit instrumentation.
5. Generated candidate modules contain functional segment calls and no direct
   mutable AITER forward calls. Compiler graph breaks are recorded and rejected.

Run only this one-block correctness study, with a 900-second remote deadline.
No timing, full-model quality or production readiness claim. A failed gate stops
the study. Record any harness correction separately. Leave raw data remote,
record local findings, restore remote main and stop before the next experiment.

Reference documentation:
[PyTorch custom operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)
and [functional/fake contracts](https://docs.pytorch.org/docs/2.14/library.html#torch.library.custom_op).

Harness revision 1: the initial attempt c8a43dce passed seven setup checks but
stopped before attention because generic deepcopy cannot construct a vLLM
parameter. Copy the block structure while sharing the same read-only loaded
parameters and buffers through deepcopy's memo. Candidate and criteria are
unchanged. Retain the original output separately and use a new output directory.
