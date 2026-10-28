# Current-main ROCm stage profile

User direction: profile with ROCm attention before choosing the next optimization.
Baseline only. Main e6a7e63d, unchanged model implementation; no P1 or direct-output
prototype. Explicit FLASH_ATTN bound to installed AITER, with SDPA calls forbidden.
Single MI300X/gfx942, BF16, batch one, TP1/SP1, both CFG branches, actual checkpoint
and the existing four-reference encoded context. Regional compilation remains on.

Use the production diffuse loop for the first three entries of the saved 40-step
schedule. This covers all-layer prefill, first decode with graph setup, and one
steady decode step. The scheduler is initialized with the complete 40-step
schedule. Do not substitute a newly generated three-step schedule. Encoder, VAE,
HTTP and final image quality are outside this diagnostic scope.

Run once to warm compilation, twice without a profiler for stage event intervals,
then once under torch.profiler with CPU/GPU activities and shapes. Fresh request
cache and graphs each run, as in production; prior-request cleanup is outside
the timed interval. Record initial compilation separately. Never use profiler
durations as unprofiled request latency. Do not run a 40-step campaign.

Outer prediction ranges identify prefill/capture-decode/steady-decode and CFG
branch. Graph setup has a nested annotation including its two warmups. Only the
profiled run adds block ranges outside the already-compiled block functions;
no hooks are inserted inside compiled operations. Record any extra compilation.

Require resolved AITER binding, all 32 loaded blocks, six branch calls, two graph
captures, finite outputs and absence of SDPA. Check repeated final-latent bytes;
record any baseline drift as a diagnostic limitation. Capture event timings at
stage boundaries without extra synchronization inside the denoising loop.

Analyze the completed trace on the remote host. Attribute device kernels via
actual HIP launch correlation and enclosing CPU annotations, not by comparing
asynchronous GPU timestamps with CPU ranges. Report unassigned events. Separate
kernel sums, device busy unions, elapsed spans and host gaps. Map GEMM order to
the model source and retain uncertainty where a source mapping is not established.

Record the dominant costs, avoidable work supported by code, and the smallest
next test. Hardware counters or instruction tracing are deferred until a specific
kernel question remains. Keep raw artifacts remote and write local findings.
Maximum process time: 15 minutes including weights and compilation. Report before
starting another experiment.
