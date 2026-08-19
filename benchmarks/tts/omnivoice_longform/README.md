# OmniVoice long form benchmark

The benchmark compares the reference OmniVoice implementation with vLLM-Omni for long English text. It measures speech coverage, word error rate, generation latency, real time factor, and vLLM serving throughput.

The benchmark runs the reference model first. It then starts one vLLM-Omni server and tests concurrency 1 by default. After the server exits, Whisper transcribes the saved concurrency 1 audio on the GPU.

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

At concurrency 1, both backends warm all eight mode and length cells before their measured requests. Each additional vLLM serving sweep also warms all eight cells and sends at least one full concurrency wave. OmniVoice uses batch size 1, so concurrency 2 and 4 measure queueing and saturation rather than request batching.

## Prompt source

The benchmark uses the `long_tts_eval_en` split of Hugging Face [`wcy1122/Long-TTS-Eval`](https://huggingface.co/datasets/wcy1122/Long-TTS-Eval), pinned to revision `fccd057bd96982e13e59c509d2921538e00a17d1`. It deterministically selects 25 source rows across the dataset's content categories. Each source row produces four nested prefixes ending at sentence boundaries. This creates 100 prompts while controlling for topic and opening context across the length buckets.

The selection rules are checked in as [selection.toml](selection.toml). The script downloads and resolves the dataset before model generation starts, then saves the exact prompts and hashes as `prompts.json` in the result directory. Dataset download time is not included in generation latency.

## Requirements

Run the benchmark inside a container that already has ROCm PyTorch and vLLM-Omni. The script installs `omnivoice==0.2.1` and the packages needed for scoring with `--no-deps`, so it does not replace the container's PyTorch.

## Run

Run the full comparison from the repository root with one command:

```bash
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh
```

The script uses the container's `python` and `vllm` commands. It installs the pinned benchmark packages, downloads the pinned dataset and model revisions, runs both backends, runs Whisper, and saves the result under `benchmarks/tts/omnivoice_longform/results/<timestamp>/`. The script prints the full result path before it starts generation and again when it finishes.

Use the small run to select 10 prompts from each word bucket. The small run has 40 prompts. It makes 80 reference requests and 80 vLLM requests at each concurrency because every prompt runs in one-shot and chunked modes.

```bash
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh --small
```

Concurrency 1 is always included because its audio is used for the quality comparison. Repeat `--concurrency` to add more serving sweeps.

```bash
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh \
    --concurrency 2 \
    --concurrency 4
```

The defaults use GPU 0, float32 generation for both implementations, Whisper large v3 in float32, seed 42, and serving concurrency 1. The OmniVoice and Whisper model revisions are pinned so rerunning the benchmark does not silently change either model.

The following environment variables change the run:

```bash
GPU_INDEX=0 \
MODEL_REVISION=c5fdb5ccb189668d56333f77ba2629f4cd7535f4 \
WHISPER_DTYPE=float32 \
WHISPER_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1 \
SEEDS="42" \
OUTPUT_DIR=/path/to/results \
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh --concurrency 4
```

Set `BENCH_PYTHON` or `VLLM_BIN` only when the container uses different command names or paths.

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
