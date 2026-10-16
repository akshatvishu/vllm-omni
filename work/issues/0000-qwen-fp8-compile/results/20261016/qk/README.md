# Current shared QK/RoPE on MI300X

Keep the accepted local operation. The shared candidate passes its operator
tolerance on twelve captures, but was slower in every paired public-call
measurement. No request timing or image-quality screen followed.

## Source and workload

PR #7594 head `7e052ecd6dd5858d9770b1965036d5c4c6dce36a` was tested through
branch commit `b341d236bfbaf36ab4f99537f0082fc4550bcd62`. The only production
source change opens the common availability guard narrowly to gfx942 BF16
interleaved D128. The reference revision was Omni `f6a0b5f9`.
Torch was 2.13 git 733fca1, Triton 3.8.0 git 669b31ac, vLLM 0.31.0+rocm723,
and HIP 7.2.53211. Twelve historical captures cover blocks 0/29/59, steps
0/49, and both CFG branches. Image length is 4096; text lengths are 21 and 5.

## Results

| Text tokens | Cache | Accepted public call, us | Shared including table, us | Paired slowdown |
| --- | --- | ---: | ---: | ---: |
| 21 | Hot | 132.40 | 165.99 | 25.18% |
| 5 | Hot | 133.53 | 167.79 | 25.65% |
| 21 | Flushed | 99.53 | 103.85 | 3.55% |
| 5 | Flushed | 98.29 | 107.91 | 9.16% |

Five rotating rounds per condition used twelve observations of twenty calls,
an accepted A/A control and a 512 MiB flush. Every one of the twenty paired
candidate differences was slower. The narrowest difference is close to A/A
variation. Percentages are paired statistics, not ratios of displayed medians.
Event intervals include host submission gaps; these are not device-only times.
Eager native calls took 605 to 650 us, but that comparison cannot establish a
gain over compiled production. Table preparation is charged once per 60 blocks.

All twelve captures passed the PR's atol 0.0625/rtol 0.02 against its own
FP32-table eager reference. V was exact, outputs finite and repeatable, poisoned
outputs fully written, and compilation counters unchanged during measurement.
Against native production math, accepted Q/K changed 60 of 302,948,352 values
(maximum 0.015625); shared Q/K changed 50,132,362 (16.55%, maximum 0.0625).
The shared operation rounds rotary coefficients differently. These ROCm
differences do not establish an image defect or a CUDA result.

## Trace and cross-run limits

Two saved dispatch traces each show one `_joint_qkv_norm_rope_kernel.kd` call
at 62.301 us. Single profiled calls do not explain the flushed public-call
difference from the older run on another host. The joint kernel source is
identical at PR heads 4e4a0c53 and 7e052ecd and leaves the new helper argument
`cuda_rms_norm` at false. Its default normalization expression is unchanged.
Current flushed table charges are 0.264 to 0.283 us/block, below the older
roughly 0.97 us charge. The cross-run timing shift remains unattributed.
Compare both revisions in one session before reporting a source regression.

Evidence: [contract](contract.json), [summary](audit-summary.json),
[accuracy](accuracy.json), [timings](timings.json), [table packing](packing.json),
[charged pairs](charged.json), and [trace/source checks](trace-and-source-check.json).
Full traces and captured operands remain in the local archive. The shared
interface can still host a ROCm implementation; this run does not decide that
upstream integration choice.
