# MI300X ROCm recipe validation

## Conclusion

Use one command to run the suite, but run each model one at a time. `scripts/run_all_sequential.sh` is the single entry point. It starts a fresh Python process for each model, waits for it to finish, records the result, and then starts the next model. Do not load these models together on one MI300X.

The current repository has recipes for Ming Flash Omni 2.0, Ming Omni TTS, Qwen3 TTS, SenseNova U1, Stable Audio Open, and MammothModa2. OmniVoice has examples and end to end tests but no recipe.

Full Ming Flash Omni 2.0 does not pass the one card capacity check in BF16. Hugging Face reports 104B BF16 parameters, which require about 208 GB before runtime state, while one MI300X has 192 GB HBM3. Its repository is 238 GB, and the current vLLM Omni recipe uses four H100 80 GB GPUs for the thinker. The standalone Ming Flash TTS stage is a separate candidate because the current recipe runs that stage on one H100 80 GB GPU, but ROCm support has not been verified.

The following files contain the checked evidence and the validation plan.

- `existing_recipe_audit.md` lists each existing recipe, its hardware, its recorded results, and the MI300X capacity decision.
- `measurement_plan.md` lists the measurements needed before adding a ROCm section to each recipe.
- `flag_trace.md` traces every environment setting used or considered here to a consumer.
- `scripts/run_all_sequential.sh` runs all default candidates in sequence.
- `scripts/run_one.sh MODEL` runs one candidate.
- `scripts/preflight.sh` checks that the environment is ROCm on one MI300X and records software versions. It accepts the `MI300X` device name, or the `gfx942` architecture with 180 to 220 GiB of visible memory when PyTorch returns an empty device name.
- `scripts/validate_artifact.py` validates generated WAV and image files.

## Recommended run

Run inside the ROCm environment from the repository root:

```bash
./mi300x_rocm_recipe_plan/scripts/run_all_sequential.sh
```

The default suite covers Ming Omni TTS, all three Qwen3 TTS task types, SenseNova U1 text to image, Stable Audio Open, MammothModa2 Preview text to image, and OmniVoice. Stable Audio requires prior acceptance of the Hugging Face license. The suite does not download models in advance, so enough local storage and Hugging Face access must already be available.

Run the standalone Ming Flash TTS candidate separately because its checkpoint download is 238 GB and its ROCm path is not verified:

```bash
RUN_MING_FLASH_TTS=1 ./mi300x_rocm_recipe_plan/scripts/run_one.sh ming_flash_tts
```

Run the Qwen3 TTS eager and graph comparison separately:

```bash
./mi300x_rocm_recipe_plan/scripts/run_one.sh qwen3_tts_compare
```

The comparison runs CustomVoice, VoiceDesign, and Base in both modes. The eager runs use the checked ROCm override and remove `MIOPEN_FIND_MODE` from the process environment. The graph runs disable stage 1 eager mode and set `MIOPEN_FIND_MODE=FAST`. The script writes request time, stage 1 time, process time, and status to `qwen3_tts_eager_vs_miopen_fast.tsv` under the result directory. The default sequential suite still uses the checked eager mode.

List the suite without starting a model:

```bash
./mi300x_rocm_recipe_plan/scripts/run_all_sequential.sh --list
```

Results are written under `mi300x_rocm_recipe_plan/results/<timestamp>/`. Each model has a command log, one second GPU samples when `rocm-smi` or `amd-smi` is available, an elapsed time record, generated output, and an artifact validation report.

## Checked ROCm environment path

The current repository installation guide validates ROCm on `gfx942`, which is the MI300X architecture. It currently documents the vLLM `0.26.0+rocm723` wheel and the `vllm/vllm-omni-rocm:v0.26.0` image. It also requires replacing `onnxruntime` with `onnxruntime-rocm` for Qwen3 TTS. Use those versions or record the exact newer versions used for the run. The preflight rejects a CUDA PyTorch build and records the available ONNX Runtime providers.

The scripts use `python3` from the container `PATH` by default. Set `PYTHON_BIN` only when the ROCm environment exposes another Python command or interpreter path.

## What can be claimed now

Only Ming Omni TTS already has a checked MI300X ROCm recipe in this repository. The other capacity decisions mean the weights and the measured CUDA memory use are below 192 GB. They do not prove ROCm correctness or performance. Add a ROCm section to a model recipe only after its corresponding script passes on the target MI300X and the recorded output passes the checks in `measurement_plan.md`.

## Repository state used for this audit

- vLLM Omni commit: `8ecd1f6d5cc91aab8a475a861213720b336e2f65`
- Audit date: 2026-08-14
- Local Python environment: Python 3.12.10 with a CUDA build of PyTorch, so no GPU run was possible in this session.
