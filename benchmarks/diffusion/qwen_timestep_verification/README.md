# Qwen timestep transfer verification

This is an experiment against main `f6a0b5f99634ca9b1eecc6bed10328eac4d28b11`.
It does not edit installed model source and does not constitute a production PR.
The diffusion benchmark profile skill informs the fixed workload and separate
correctness, timing, and profiling runs.

## Source findings

The ordinary Qwen loop is in `models/qwen_image/cfg_parallel.py`, not the generic
CFG mixin. Each iteration expands a CPU timestep and copies it to the latent
device and dtype. The model receives that value divided by 1000. The scheduler
receives the original CPU timestep.

Step execution also has a CPU to GPU timestep transfer on this revision.
`prepare_timesteps()` does not pass a device to `retrieve_timesteps()`.
`prepare_encode()` stores the resulting CPU schedule in `state.timesteps`.
`InputBatch._prepare_timesteps()` passes the current CPU values to
`_gather_tensor_rows()`, which allocates on the input tensor device. Neither the
batch builder nor the runner moves that field to the latent device.
`denoise_step()` sends it to `_build_denoise_kwargs()`, which performs the copy.

The candidate stages the ordinary model schedule once before the loop. For step
execution it stages a separate request-owned model schedule in `state.extra`,
and selects the appropriate values using the current `input_batch.states`.
`state.timesteps`, the per-request scheduler, and scheduler input remain on CPU.
The existing BF16 cast still precedes division by 1000. All mask code is unchanged.
The step candidate still performs a GPU concatenation on each step. This experiment
only removes the repeated model timestep transfer; it does not optimize batching.

The generated `timestep-only.patch` shows the exact candidate. `sync_overlay.py`
checks original source SHA256 values before applying those changes in memory.
`sitecustomize.py` installs the import hook in spawned Python workers through
`PYTHONPATH`. The harness also installs the hook in its own process. Every run
requires worker overlay proof before it is marked complete.

## GPU commands

Run from the checked-out repository with its installed dependencies. Set `SYNC`
to this directory and `MODEL` to the pinned, downloaded Qwen-Image snapshot.
Choose an empty output directory for every run. The scripts refuse to overwrite
an existing run directory. Preserve the same attention, AITER, quantization and
other environment settings across all arms. No custom quant kernel is installed
by this experiment.

```bash
SYNC="$PWD/work/issues/0000-qwen-fp8-compile/runs/20261016-verification/sync"
MODEL="/absolute/path/to/pinned/Qwen-Image/snapshot"
export PYTHONPATH="$SYNC:$PWD${PYTHONPATH:+:$PYTHONPATH}"

python "$SYNC/run_sync.py" --model "$MODEL" --arm baseline --mode ordinary --phase correctness --output "$SYNC/results/ordinary-baseline"
python "$SYNC/run_sync.py" --model "$MODEL" --arm timestep --mode ordinary --phase correctness --output "$SYNC/results/ordinary-timestep"
python "$SYNC/run_sync.py" --model "$MODEL" --arm baseline --mode step --phase correctness --output "$SYNC/results/step-baseline"
python "$SYNC/run_sync.py" --model "$MODEL" --arm timestep --mode step --phase correctness --output "$SYNC/results/step-timestep"

python "$SYNC/compare_sync.py" --baseline "$SYNC/results/ordinary-baseline" "$SYNC/results/step-baseline" --candidate "$SYNC/results/ordinary-timestep" "$SYNC/results/step-timestep" --output "$SYNC/results/correctness.json"
python "$SYNC/compare_sync.py" --baseline "$SYNC/results/ordinary-baseline" --candidate "$SYNC/results/step-baseline" --output "$SYNC/results/mode-parity.json"

# Ordinary request latency: three fresh AB / BA / AB pairs plus independent A/A.
# Each process uses two warmups and six measured samples; all share compilation caches.
python "$SYNC/run_sync.py" --model "$MODEL" --arm baseline --mode ordinary --phase timing --output "$SYNC/results/timing-p1-a"
python "$SYNC/run_sync.py" --model "$MODEL" --arm timestep --mode ordinary --phase timing --output "$SYNC/results/timing-p1-b"
python "$SYNC/run_sync.py" --model "$MODEL" --arm timestep --mode ordinary --phase timing --output "$SYNC/results/timing-p2-b"
python "$SYNC/run_sync.py" --model "$MODEL" --arm baseline --mode ordinary --phase timing --output "$SYNC/results/timing-p2-a"
python "$SYNC/run_sync.py" --model "$MODEL" --arm baseline --mode ordinary --phase timing --output "$SYNC/results/timing-p3-a"
python "$SYNC/run_sync.py" --model "$MODEL" --arm timestep --mode ordinary --phase timing --output "$SYNC/results/timing-p3-b"
python "$SYNC/run_sync.py" --model "$MODEL" --arm baseline --mode ordinary --phase timing --output "$SYNC/results/timing-aa-1"
python "$SYNC/run_sync.py" --model "$MODEL" --arm baseline --mode ordinary --phase timing --output "$SYNC/results/timing-aa-2"
python "$SYNC/compare_sync.py" --baseline "$SYNC/results/timing-p1-a" "$SYNC/results/timing-p2-a" "$SYNC/results/timing-p3-a" --candidate "$SYNC/results/timing-p1-b" "$SYNC/results/timing-p2-b" "$SYNC/results/timing-p3-b" --output "$SYNC/results/timing.json"
python "$SYNC/compare_sync.py" --baseline "$SYNC/results/timing-aa-1" --candidate "$SYNC/results/timing-aa-2" --output "$SYNC/results/timing-aa.json"

# Separate, short profiles. Do not use these request times as benchmark samples.
python "$SYNC/run_sync.py" --model "$MODEL" --arm baseline --mode ordinary --phase profile --steps 2 --output "$SYNC/results/profile-baseline"
python "$SYNC/run_sync.py" --model "$MODEL" --arm timestep --mode ordinary --phase profile --steps 2 --output "$SYNC/results/profile-timestep"
```

Default workload: 1024 by 1024, 50 steps, CFG 4, BF16 activations with online FP8
weights, regional dynamic compile, no CPU or layerwise offload, one request at a
time. Correctness covers five prompts and distinct fixed seeds. `--prompts 2`
provides a smaller explicitly bounded run. `--eager` selects eager operation and
must be consistent between arms; it is not the default.

## Evidence and limits

Correctness writes a PNG and CPU audit tensor file per request. Startup dummy
requests are recorded separately. Each measured request records the exact audit
filename created during its synchronous generation, avoiding startup index offsets. Audits contain
final latents, model timestep values/dtypes/devices, scheduler timestep values,
and step-mode input devices. Scheduler audits assert CPU inputs. The comparator
requires bitwise identical pixels, latents, and timestep values. Any mismatch
fails the comparison rather than silently accepting an image metric threshold.
Baseline step audit should show CPU builder inputs and candidate step audit
should show GPU inputs. Both model outputs must be BF16 and identical.

Timing measures synchronous `Omni.generate()` completion, including orchestration,
text encoding, denoising, decoding, and output transfer. Image hashing and saving
happen after that interval. Correctness hooks are disabled in timing processes.
Three fresh process AB / BA / AB pairs check repeatability. An independent A/A
pair estimates variation between baseline runs. Warmups do not prove there were zero later recompiles. Check logs
and profiles for compilation before making a small latency claim. Record GPU
utilization and clocks externally around these runs.

The scope is Qwen-Image, one GPU, one active request, and the pinned software and
model revisions. The ordinary mixin is shared with editing/layered pipelines;
those pipelines are not validated by these text-to-image runs. The step schedule
selection follows batch state ordering but concurrent batch and cache-repacking
coverage is outside this harness. Review traces to confirm the expected reduction
in timestep copies and synchronizations; wall time alone does not establish the
cause or a general speedup.

Local preparation completed source hash generation and Python AST, CLI help, and shell syntax checks only.
No GPU correctness, model execution, timing, or trace result is claimed here.

## Remote launcher

`remote/run_sync.sh` runs against the pinned container and checked-out sync branch.
It requires `SYNC_EXPECTED_SHA` to identify the exact pushed harness commit. Run
`bash run_sync.sh diagnostic diagnostic-1` first for ordinary baseline correctness
with two prompts and two steps, including a self-comparison of audit associations.
Then run `bash run_sync.sh full full-1`. Every output tag must be new. The launcher
checks the GPU is idle, the container identity, branch and source revision, model
shards, repository import path, and source hashes. Shared Inductor and Triton caches
live outside the repository. Dependencies come from the isolated staging directory;
the launcher does not install packages or edit installed source.
