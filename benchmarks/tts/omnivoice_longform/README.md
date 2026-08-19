# OmniVoice long form benchmark

The benchmark compares the official OmniVoice package with the vLLM-Omni implementation on long English text. It measures how much of the input text appears in the generated speech, word error rate, generation time, real time factor, and vLLM serving throughput.

## Quick start

Run the full benchmark from the repository root inside a container that has ROCm PyTorch and vLLM-Omni installed:

```bash
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh
```

The default run selects 100 prompts. Each prompt runs in one shot mode and chunked mode with both backends, which produces 400 measured generation cases at concurrency 1.

Use `--small` for 10 prompts in each word count group. The small run selects 40 prompts and produces 160 measured generation cases at concurrency 1.

```bash
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh --small
```

Concurrency 1 is always included because Whisper uses its saved audio for the quality comparison. Repeat `--concurrency` to add vLLM serving sweeps at other concurrency levels.

```bash
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh \
    --concurrency 2 \
    --concurrency 4
```

The script prints the result directory before generation starts and after the benchmark finishes. By default, results are saved under `benchmarks/tts/omnivoice_longform/results/<timestamp>/`.

## What the script runs

The script runs each stage in this order:

1. It installs `omnivoice==0.2.1`, `jiwer==4.0.0`, and `pydub==0.25.1` with `--no-deps`.
2. It downloads the pinned Long TTS Eval data and saves the selected prompts and their hashes.
3. It runs the official OmniVoice package and saves its audio.
4. It starts one vLLM-Omni server, runs every requested concurrency sweep, and then stops the server.
5. It loads Whisper on the GPU, transcribes the saved concurrency 1 audio, and writes the quality summary.

The official implementation and the vLLM-Omni server do not run at the same time. Whisper starts only after the vLLM-Omni server has stopped, so Whisper memory does not affect generation measurements.

## Requirements

Run the benchmark in a container that already provides ROCm PyTorch, torchaudio, vLLM, vLLM-Omni, and the OmniVoice runtime dependencies. The script uses the container's `python` and `vllm` commands. It installs the pinned benchmark packages with `--no-deps`, so it does not replace the container's PyTorch packages.

The run also needs `bash`, `curl`, internet access to Hugging Face, and enough disk space for the models, dataset, audio, and JSON results. One MI300X with 192 GB of VRAM is sufficient for the default concurrency 1 run.

## Command options

| Option | Default | Meaning |
| --- | --- | --- |
| `--small` | Off | Select 10 source rows instead of 25. Each row produces one prompt in each of the four word count groups. |
| `--concurrency N` | Concurrency 1 only | Add concurrency `N` to the vLLM serving sweeps. The option can be repeated. |
| `-h`, `--help` | | Print the command usage. |

The script rejects zero, negative, missing, and repeated concurrency values.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU_INDEX` | `0` | GPU exposed through `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`. |
| `PORT` | `8091` | Port used by the temporary vLLM-Omni server. |
| `MODEL` | `k2-fsa/OmniVoice` | Hugging Face model name or local model directory. |
| `MODEL_REVISION` | `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` | Pinned OmniVoice model revision. |
| `WHISPER_MODEL` | `openai/whisper-large-v3` | Whisper model used for transcription. |
| `WHISPER_REVISION` | `06f233fe06e710322aca913c1bc4249a0d71fce1` | Pinned Whisper model revision. |
| `WHISPER_DTYPE` | `float32` | Whisper data type. Accepted values are `float32`, `float16`, and `bfloat16`. |
| `SEEDS` | `42` | Space separated generation seeds, such as `"42 123"`. |
| `OUTPUT_DIR` | `results/<timestamp>` | Directory used for audio and result files. |
| `BENCH_PYTHON` | `python` | Python command used by the script. |
| `VLLM_BIN` | `vllm` | vLLM command used by the script. |

Example:

```bash
GPU_INDEX=0 \
PORT=8095 \
SEEDS="42" \
OUTPUT_DIR=/path/to/results \
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh --small --concurrency 2
```

## Test matrix

The default quality comparison contains 400 measured outputs:

| Dimension | Values |
| --- | --- |
| Backends | Official OmniVoice package and vLLM-Omni |
| Prompt distribution | 25 near 120 words, 25 near 200 words, 25 near 300 words, and 25 from 400 to 600 words |
| Generation modes | One shot and 15 second chunks |
| Seeds | 42 |
| Batch size | 1 |
| Concurrency used for quality | 1 |

Each backend warms one representative request from all eight generation mode and word count combinations before measuring concurrency 1. Each extra vLLM serving sweep repeats that coverage and sends at least one full concurrency wave before measurement.

OmniVoice uses batch size 1. Concurrency 2 and higher measure request queueing and server saturation, not model request batching.

With the default seed, each backend receives 200 measured requests at concurrency 1. Each extra vLLM concurrency sweep receives another 200 measured requests. The small run uses 80 measured requests per backend or vLLM concurrency sweep. Extra seeds multiply these counts.

## Prompt source

The benchmark uses the `long_tts_eval_en` split of Hugging Face [`wcy1122/Long-TTS-Eval`](https://huggingface.co/datasets/wcy1122/Long-TTS-Eval), pinned to revision `fccd057bd96982e13e59c509d2921538e00a17d1`.

The selector chooses 25 source rows across the dataset categories. Each source row produces four prefixes that end at sentence boundaries, with one prefix near each target length. The four prompts from one row share the same topic and opening text, so the comparison changes length without changing the start of the passage.

The selection rules are stored in [selection.toml](selection.toml). Dataset download and prompt selection happen before model generation, so their time is not included in generation latency. The exact prompts, source row IDs, word counts, and text hashes are saved in `prompts.json`.

## Generation modes

One shot mode sets `audio_chunk_threshold=1000000`. The large finite value disables chunking without using infinity.

Chunked mode sets `audio_chunk_threshold=0` and `audio_chunk_duration=15`. Both implementations receive the same values.

## Metrics

Whisper transcripts each successful audio file and compares the transcript with the source text. The quality summary reports the following values for every backend, generation mode, and word count group:

```text
WER = (substitutions + deletions + insertions) / reference words
coverage = 1 - (substitutions + deletions) / reference words
RTF = generation time / generated audio duration
```

Insertions increase WER but do not reduce coverage. The evaluator uses word sequence alignment from `jiwer`, so repeated words are counted by their position instead of by set membership. A lower WER and RTF are better, while a higher coverage is better. An RTF below 1 means generation was faster than the audio duration.

The serving summary also records successful and failed request counts, request throughput, generated audio throughput, latency percentiles, and RTF percentiles. Failed requests do not count toward successful request or audio throughput.

## Whisper transcription

The evaluator uses `openai/whisper-large-v3`, 16 kHz mono audio, English transcription, and timestamp based long audio generation. It passes `truncation=False`, so audio after Whisper's 30 second input window is not discarded.

The evaluator prints when Whisper is loading, when transcription starts, and one progress line for every restored, successful, or failed case.

## Resume an interrupted run

Set `OUTPUT_DIR` to the same directory and rerun the same command. Reference generation and Whisper transcription save progress after every case. The vLLM benchmark saves progress after every completed concurrency sweep. Failed cases remain in the result files instead of causing the completed work to be discarded.

The benchmark rejects a resume when the model revision, Whisper revision, dependency versions, repository contents, run settings, or prompt manifest do not match the original run. The check prevents results from different code or model versions from being combined in one summary.

## Output files

Each run creates the following files:

```text
results/<timestamp>/
├── prompts.json
├── run_metadata.json
├── reference/
│   ├── generation.jsonl
│   ├── summary.json
│   └── *.wav
├── vllm-omni/
│   ├── generation.jsonl
│   ├── serving.jsonl
│   ├── serving_summary.json
│   ├── server.log
│   └── *.wav
└── evaluation/
    ├── evaluated.jsonl
    ├── summary.json
    └── summary.md
```

`generation.jsonl` contains one row for every measured case. `serving.jsonl` contains every vLLM concurrency sweep, while `vllm-omni/generation.jsonl` contains the concurrency 1 rows used for quality evaluation. `evaluation/summary.md` is the main quality table, and `vllm-omni/serving_summary.json` contains the serving metrics.

`run_metadata.json` records the pinned model revisions, package versions, GPU identity, prompt manifest hash, repository state, and run settings.

## GPU memory measurement

The benchmark records the peak GPU memory reserved by PyTorch for every successful generation request. The official OmniVoice runner reads the value in its process. The vLLM-Omni server returns the value in the `X-Peak-Memory-MB` response header, and the benchmark converts it to GiB. PyTorch uses the same API with the ROCm backend, so the measurement does not depend on an `amd-smi` command being present in the container.

The value includes PyTorch's reserved memory pool. It does not include GPU memory allocated outside PyTorch.

## Compare GPU retention with CPU copies

Use the memory A/B runner to measure the effect of copying each decoded chunk to CPU instead of keeping every decoded chunk on the GPU until generation finishes:

```bash
bash benchmarks/tts/omnivoice_longform/run_memory_ab.sh
```

The runner tests one Long TTS Eval source row by default. The row produces four nested prompts at about 120, 200, 300, and 500 words, so the longest request exercises the path with the most retained chunks. Each variant warms all four lengths before measurement. PyTorch can keep reserved memory after a warmup, so use the overall maximum to judge GPU memory savings rather than treating each word count row as an independent cold process.

Both variants use the same committed vLLM-Omni revision. The runner uses [prepare_gpu_retention_baseline.py](prepare_gpu_retention_baseline.py) to change the baseline worktree so it keeps decoded chunks on the GPU. The candidate worktree uses the normal code, which copies each decoded chunk to CPU. The runner starts a fresh server for each worktree and runs the variants one after the other on the same GPU.

The A/B run measures only chunked vLLM-Omni requests at concurrency 1. It does not run the official OmniVoice package or Whisper, and it does not save WAV files. It compares the SHA256 hashes of the returned WAV data so the report shows whether both variants produced identical audio. Results are saved under `results/memory-ab/<timestamp>/`. The main report is `comparison.md`, and `comparison.json` contains the same values in a machine readable form.

Use more source rows when you need a more stable latency result:

```bash
bash benchmarks/tts/omnivoice_longform/run_memory_ab.sh --samples 3
```

Set `TARGET_REF` or pass `--ref` to test another committed revision. The baseline transform stops with an error if that revision does not contain the expected CPU-copy code.
