# Measurements required for each ROCm recipe

## Record for every model

Every run must save the following information before a ROCm recipe claims support.

- Record the vLLM Omni commit, vLLM version, Python version, PyTorch version, ROCm or HIP version, container tag or image digest, operating system, and GPU name.
- Record the exact model revision when possible, exact command, prompt, seed, output settings, and whether the first run includes model download or kernel compilation.
- Record model load time, first request time, at least three warm request times, peak HBM use, idle HBM after load, GPU use, power, and temperature.
- Save the generated artifact and its basic validity checks. A successful process with an empty or invalid file is a failed run.
- Save the full log. Record fallbacks, compilation errors, warnings about unsupported kernels, and whether eager mode or a platform override was active.
- Run one negative or fallback check. For example, confirm that Qwen3 TTS selects the ROCm eager decoder override, or confirm that a missing gated model token fails before a recipe result is recorded.

The supplied scripts collect the environment, full command log, elapsed wall time, one second GPU samples, and one smoke output. A publishable performance table needs a separate loaded process that handles at least four requests. Treat the first request after load as cold and report the median of the next three requests as warm. Restart the process only when measuring load time or cold start time.

## Ming Omni TTS

Record basic synthesis and one reference conditioned case. Save output sample rate, channels, duration, peak amplitude, RMS, time to first audio for streaming, total generation time, real time factor, stage durations, and peak memory. The existing example already writes a JSON manifest with stage duration and peak memory fields, so preserve that manifest in the recipe evidence.

## Qwen3 TTS

Run CustomVoice, VoiceDesign, and Base in separate processes because each task uses a different checkpoint. Record sample rate, duration, RMS, time to first audio for streaming, total time, real time factor, and peak memory for each checkpoint. Confirm in the log or resolved config that ROCm changes stage 1 `enforce_eager` from false to true. Record whether `onnxruntime-rocm` exposes `ROCMExecutionProvider` for the tokenizer paths that need it.

## SenseNova U1

Use the existing 1536 by 2720, 50 step, seed 42 workload so the MI300X result can be compared with the current H200 row. Record model load time, total generation time, peak allocated and reserved HBM, output dimensions, and the exact `think`, CFG, timestep shift, and epsilon settings. Save the image for visual review. Add image to image and understanding results only after the text to image baseline passes.

## Stable Audio Open

Use the existing 10 second, 50 step, seed 42, TeaCache workload. Record load time, generation time, real time factor, peak HBM, sample rate, channel count, exact duration, RMS, and peak amplitude. Record that the Hugging Face license was accepted, but never write the token into a log or recipe.

## MammothModa2

Start with Preview text to image at 1024 by 1024, 50 steps, and seed 42. Record AR load time, DiT load time, total generation time, peak HBM, output dimensions, generated visual token count, and the exact stage memory split. Run Dev understanding and Dev text to image separately. The skipped end to end test for issue 3201 is a known evidence gap, so a valid output image is required before adding a ROCm claim.

## OmniVoice

Record automatic voice generation first. Then record fixed seed repeatability, different seed output, reference audio cloning without text, and reference audio cloning with text. Save sample rate, duration, RMS, total time, real time factor, peak HBM, and whether Triton kernels or PyTorch fallbacks ran. A new recipe must include the exact environment because no current recipe exists.

## Ming Flash Omni 2.0

Do not test the full model on one MI300X. If the standalone TTS stage is tested, record checkpoint disk use, loaded HBM, load time, output sample rate, duration, RMS, total time, real time factor, and whether the inner CFM graph ran or fell back. Do not convert a standalone TTS success into a claim for the thinker, image stage, or full any to any model.
