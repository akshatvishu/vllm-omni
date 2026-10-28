# ROCm profile compatibility revision

The original protocol and failed run remain preserved. Full-block regional
compilation failed in installed Torch's `decompose_auto_functionalized` V2 handler,
at `post_grad.py:1570`, before any completed timing or trace.

Installed vLLM `config/compilation.py:924-926` supplies
`enable_auto_functionalized_v2=False` by default for its custom-op compatibility.
Omni's `diffusion_model_runner.py:255-264` supplies only graph-related compile
options for this model. This revised probe explicitly inherits the single vLLM
functionalization option. It preserves AITER attention, global AITER enablement,
all model source, precision, shapes, graph policy and timing boundaries.

This is a main-source profile with an explicit compiler compatibility setting,
not a successful profile of the unchanged default compiler configuration. Do
not attribute the failed V2 decomposition to a specific AITER operator without
a smaller reproducer. Do not claim this configuration is a qualified production
fix merely because profiling succeeds.

All other original protocol requirements apply. The source-backed GEMM parser
also corrects the block count to five: QKV, attention output, two SwiGLU input
projections and SwiGLU output. It combines the independent SwiGLU input pair.
This parser correction does not alter GPU execution.
