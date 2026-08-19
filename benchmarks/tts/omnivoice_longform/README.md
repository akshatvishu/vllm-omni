# OmniVoice long form benchmark

The benchmark compares the reference OmniVoice implementation with vLLM-Omni for long English text. It measures speech coverage, word error rate, generation latency, real time factor, and vLLM serving throughput.

The benchmark runs the reference model first. It then starts one vLLM-Omni server and tests concurrency 1, 2, and 4. After the server exits, Whisper transcribes the saved concurrency 1 audio on the GPU.

## Test matrix

The quality comparison contains 400 measured outputs:

| Dimension | Values |
| --- | --- |
| Backends | Reference OmniVoice and vLLM-Omni |
| Prompt distribution | 25 near 120 words, 25 near 200, 25 near 300, and 25 from 400 to 600 words |
| Generation modes | One shot and 15 second chunks |
| Seeds | 42 |
| Batch size | 1 |
| Concurrency | 1 |

The reference backend warms both generation modes. Each vLLM serving sweep warms all eight mode and length cells and sends at least one full concurrency wave before its 200 measured requests. OmniVoice uses batch size 1, so concurrency 2 and 4 measure queueing and saturation rather than request batching.

## Prompt source

The benchmark uses the `long_tts_eval_en` split of Hugging Face [`wcy1122/Long-TTS-Eval`](https://huggingface.co/datasets/wcy1122/Long-TTS-Eval), pinned to revision `fccd057bd96982e13e59c509d2921538e00a17d1`. It deterministically selects 25 source rows across the dataset's content categories. Each source row produces four nested prefixes ending at sentence boundaries. This creates 100 prompts while controlling for topic and opening context across the length buckets.

The selection rules are checked in as [selection.toml](selection.toml). The script downloads and resolves the dataset before model generation starts, then saves the exact prompts and hashes as `prompts.json` in the result directory. Dataset download time is not included in generation latency.

## ROCm setup

Use an existing ROCm PyTorch and vLLM-Omni environment. Do not install the reference project's dependencies as a group because its lock file selects CUDA PyTorch wheels.

The runner installs the selected local reference package before generation:

```bash
.venv/bin/python -m pip install --no-deps -e work/repos/k2-fsa/OmniVoice
```

The environment must already contain the packages other than PyTorch that OmniVoice requires, including Transformers, Accelerate, pydub, NumPy, SoundFile, and torchaudio.

Do not enable the reference FlashInfer path on ROCm.

## Run

Run the full comparison from the repository root:

```bash
benchmarks/tts/omnivoice_longform/run_benchmark.sh
```

The defaults use GPU 0, float32 generation for both implementations, Whisper large v3 in float32, seed 42, and serving concurrency 1, 2, and 4. The OmniVoice and Whisper model revisions are pinned so rerunning the benchmark does not silently change either model.

The following environment variables change the run:

```bash
GPU_INDEX=0 \
MODEL_REVISION=c5fdb5ccb189668d56333f77ba2629f4cd7535f4 \
WHISPER_DTYPE=float32 \
WHISPER_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1 \
SEEDS="42" \
CONCURRENCIES="1 2 4" \
OUTPUT_DIR=/path/to/results \
benchmarks/tts/omnivoice_longform/run_benchmark.sh
```

Set `REFERENCE_REPO` when the original OmniVoice checkout is not under `work/repos/k2-fsa/OmniVoice`. Set `BENCH_PYTHON` and `VLLM_BIN` when the container uses a different virtual environment.

Rerun the same command with the same `OUTPUT_DIR` to resume. Reference generation and Whisper evaluation checkpoint after every case. vLLM-Omni checkpoints after every completed concurrency sweep. Failed cases are preserved in the output instead of discarding the rest of the run. The resolved prompt manifest and fingerprinted run metadata are immutable, so a changed model revision, Whisper revision, dependency set, repository state, or manifest is rejected instead of mixed with existing results.

## Generation modes

One shot generation uses `audio_chunk_threshold=1000000`. The large finite value disables chunking without relying on infinity conversion.

Chunked generation uses `audio_chunk_threshold=0` and `audio_chunk_duration=15`. The same values are passed to both implementations.

## Whisper and scoring

The evaluator uses `openai/whisper-large-v3`, 16 kHz mono audio, English transcription, and timestamp based sequential long form generation. The processor receives `truncation=False`, so it does not discard audio after Whisper's 30 second input window.

The evaluator reports these metrics:

```text
WER = (substitutions + deletions + insertions) / reference words
coverage = 1 - (substitutions + deletions) / reference words
```

Insertions increase WER but do not reduce coverage. The evaluator uses sequence alignment from `jiwer`, so repeated words are counted correctly.

## Outputs

Each run creates these files under the selected output directory:

```text
reference/generation.jsonl
reference/summary.json
prompts.json
run_metadata.json
vllm-omni/generation.jsonl
vllm-omni/serving.jsonl
vllm-omni/serving_summary.json
vllm-omni/server.log
evaluation/evaluated.jsonl
evaluation/summary.json
evaluation/summary.md
```

The reference and concurrency 1 vLLM directories also contain one WAV file per case. Whisper runs only after both generation processes have exited, so ASR memory use does not affect generation measurements.

`run_metadata.json` records the source revisions, package versions, GPU identity, manifest hash, and run settings. Reference peak memory is recorded per request. vLLM server peak memory is not measured by this script because ROCm monitoring commands differ across container images. Validate the vLLM memory benefit separately on the MI300X with the container's ROCm monitoring tool.
