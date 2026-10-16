# Qwen timestep-only preload result

All ten correctness comparisons passed exactly. Standalone request latency is
unresolved under the frozen rule, so keep the change parked as a speed
optimization. The mask change is excluded from this experiment.

The checked-out harness commit was `0272d879812ca7237fb09ade31a7c01a7e216bdf`,
based on Omni `f6a0b5f99634ca9b1eecc6bed10328eac4d28b11`. The existing
container's AITER/vLLM dependencies were used on one MI300X. Installed source
was unchanged; source-hashed overlays applied the candidate in memory.
Qwen-Image snapshot `75e0b4be04f60ec59a75f475837eced720f823b6` ran at 1024
square, 50 steps, CFG4, BF16 activations and online FP8 weights, with regional
dynamic compilation and shared prewarmed compiler caches.

## Correctness

Five prompts/seeds in each of ordinary and step mode matched baseline exactly
for actual PNG pixels, latent bytes, and all fifty model and scheduler
timesteps. Both baseline modes also matched. Startup dummy requests were
excluded through explicit audit-file association. Step-mode builder inputs
changed from fifty CPU values to fifty GPU values; scheduler inputs stayed
on CPU. This does not cover concurrent batching, offload, editing, layered
models, multi-GPU execution, or fresh empty compiler caches.

## Request timing

Each process used two warmups and six measured requests. Three fresh-process
pairs ran in AB, BA, AB order. Positive savings mean faster candidate execution.

| Pair | Baseline median, ms | Candidate median, ms | Saving, ms |
| --- | ---: | ---: | ---: |
| 1 | 14165.897 | 14172.351 | -6.454 |
| 2 | 14193.777 | 14141.776 | 52.000 |
| 3 | 14139.690 | 14174.744 | -35.054 |

The A/A pair differed by 28.599 ms (0.2017%). Median paired saving was
-6.454 ms (-0.0456%), with disagreeing signs. The frozen decision is unresolved.
No extra rounds were added. Timing includes synchronous generate completion
and excludes subsequent image saving/hashing. Correctness hooks were off.
All timing images matched. Only ordinary mode was timed; step mode has
correctness evidence. This baseline is current main, not the historical
accepted optimization stack.

## Separate two-step profiles

Correlated CPU and GPU events identify two baseline 2-byte timestep HtoD
copies versus one candidate 4-byte schedule copy. Whole-request HtoD calls
fall from twelve to eleven. Explicit stream synchronizations remain six,
and both traces include one device synchronization from profiler stop.
These instrumented observations confirm copy removal, not a request gain.
Both profiler exports completed despite an external-init callback warning;
GPU memcpy lane coverage is incomplete, so the audit uses HIP API metadata
and correlated kernel events. Profiler request times are excluded from timing.

Evidence: [contract](contract.json), [all comparisons and samples](results.json),
[profile summary](profile-summary.json). Images, tensor audits and full traces
remain in the local archive. No new request speedup is claimed.
