# OmniVoice long form benchmark

Use this benchmark to compare the official OmniVoice package with the vLLM-Omni OmniVoice server on long English text. Both implementations receive the same prompts, model revision, seeds, and chunk settings.

The comparison covers the following behavior:

* It measures whether chunking helps OmniVoice speak all of a long prompt.
* It checks whether vLLM-Omni matches the official implementation in speech coverage and word error rate.
* It compares generation time, real time factor, throughput, failures, and peak PyTorch GPU memory.

## Quick start

Run the benchmark from the vLLM-Omni repository root inside your ROCm container. The script starts and stops the vLLM-Omni server, so you do not need to start a server first.

Start with the small run:

```bash
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh --small
```

The small run does the following work:

* It selects 10 source passages and creates 40 prompts, with 10 prompts in each word count group.
* It generates each prompt in `one_shot` and `chunked` mode with the official package. The official package therefore runs 80 measured requests.
* It generates the same 80 cases through the vLLM-Omni server at concurrency 1.
* If every generation succeeds, it saves 160 measured audio files and uses Whisper to transcribe them. Failed generations are recorded and are not sent to Whisper.
* It excludes warmup requests from every reported measurement.

The script prints the result directory when the run starts and again when it finishes. Results are saved under `benchmarks/tts/omnivoice_longform/results/<timestamp>/` unless you set `OUTPUT_DIR`.

Run the full comparison with:

```bash
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh
```

The full run selects 25 source passages and creates 100 prompts. Each backend runs 200 measured cases at concurrency 1, which gives 400 measured cases. Whisper transcribes every case that produced valid audio.

## What is compared

| Backend | How it runs | Measured concurrency | Saved audio | Purpose |
| --- | --- | ---: | --- | --- |
| Official OmniVoice | Calls `OmniVoice.generate()` from `omnivoice==0.2.1` | 1 | Every measured request | Reference quality and performance |
| vLLM-Omni | Sends `POST /v1/audio/speech` requests to a temporary server | 1 by default | Concurrency 1 only | Quality, performance, and serving behavior |

The official runner finishes before the vLLM-Omni server starts. The server stops before Whisper loads. The three GPU workloads do not run at the same time, so one workload does not consume GPU memory while another workload is measured.

The official runner loads the pinned model revision with float32. The vLLM-Omni server loads the same revision through its normal OmniVoice configuration. Both backends use batch size 1 and seed 42 by default.

## Prompts

The benchmark downloads the `long_tts_eval_en` split of [`wcy1122/Long-TTS-Eval`](https://huggingface.co/datasets/wcy1122/Long-TTS-Eval) at revision `fccd057bd96982e13e59c509d2921538e00a17d1`.

The selector chooses source passages across the dataset categories. It creates four prefixes from each source passage:

| Group | Allowed words | Target words |
| --- | ---: | ---: |
| `words_120` | 100 to 140 | 120 |
| `words_200` | 180 to 220 | 200 |
| `words_300` | 275 to 325 | 300 |
| `words_400_plus` | 400 to 600 | 500 |

Each prefix ends at a sentence boundary. The four prompts from one source passage have the same opening and topic, so prompt length changes without also changing the start of the passage. Selection is deterministic and uses seed 6333.

The script downloads and selects the prompts before generation timing starts. It saves the selected text, source row IDs, word counts, and text hashes in `prompts.json`. The full selection rules are in [selection.toml](selection.toml).

## Generation modes

Every prompt runs once in each mode:

| Mode | `audio_chunk_threshold` | `audio_chunk_duration` | Meaning |
| --- | ---: | ---: | --- |
| `one_shot` | `1000000` | `15` | The threshold is high enough to disable chunking for the benchmark prompts. |
| `chunked` | `0` | `15` | The model splits long generation into about 15 second audio chunks. |

Both backends receive the same values. The large finite threshold in `one_shot` mode avoids passing infinity through request serialization.

## Optional concurrency sweeps

Concurrency 1 is always included because its audio is used for the quality comparison. Add one or more vLLM-Omni concurrency sweeps with repeated `--concurrency` options:

```bash
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh \
    --small \
    --concurrency 2 \
    --concurrency 4
```

In this example, the official package still runs 80 measured requests. vLLM-Omni runs 80 requests at concurrency 1, another 80 at concurrency 2, and another 80 at concurrency 4. Whisper still transcribes only the 160 official and concurrency 1 audio files.

OmniVoice uses batch size 1. Higher concurrency sends several client requests at the same time, but the OmniVoice stage processes requests with batch size 1. Higher concurrency therefore measures queueing and server saturation, not request batching.

## What the script does

The command runs these steps in order:

1. It installs `omnivoice==0.2.1`, `jiwer==4.0.0`, and `pydub==0.25.1` with `pip install --no-deps`.
2. It downloads the pinned dataset and writes the prompt manifest.
3. It records the model revisions, package versions, GPU, repository state, prompt manifest hash, seeds, and concurrency settings.
4. It loads the official OmniVoice model, warms all eight mode and word count combinations, and runs the reference cases.
5. It starts the vLLM-Omni server and waits for its health endpoint.
6. It warms and measures each requested vLLM-Omni concurrency level.
7. It stops the server, loads Whisper on the GPU, transcribes the concurrency 1 audio, and writes the quality summary.

For each concurrency level, warmup covers all eight combinations of generation mode and word count group. When concurrency is greater than eight, the benchmark sends enough warmup requests to fill at least one concurrency wave. Warmups do not appear in measured request counts or result rows.

## Metrics

Whisper transcribes every successful concurrency 1 audio file. The evaluator normalizes the source text and transcript, aligns their word sequences with `jiwer`, and reports metrics for each backend, mode, and word count group.

| Metric | Definition | Better value |
| --- | --- | --- |
| Coverage | `1 - (substitutions + deletions) / reference words` | Higher |
| Word error rate | `(substitutions + deletions + insertions) / reference words` | Lower |
| Latency | Time from starting generation or sending the request until audio is ready | Lower |
| Real time factor | `generation latency / generated audio duration` | Lower |
| Request throughput | Successful requests divided by sweep wall time | Higher |
| Audio throughput | Generated audio seconds divided by sweep wall time | Higher |
| Peak reserved GPU memory | Maximum memory held by the PyTorch allocator during a request | Lower |
| Failure rate | Failed cases divided by total cases | Lower |

Insertions increase word error rate but do not reduce coverage. Coverage therefore answers whether the model spoke the source words, while word error rate also penalizes extra words. An RTF below 1 means the model generated audio faster than playback time.

The quality summary reports generation failures and transcription failures together with metric means and standard deviations. The vLLM-Omni serving summary reports generation failures, overall p50, p90, and p99 latency and RTF, plus successful request and audio throughput. Failed requests do not count toward successful throughput.

The evaluator uses `openai/whisper-large-v3` with 16 kHz mono audio, English transcription, timestamps, and `truncation=False`. The timestamp mode allows Whisper to transcribe audio longer than its 30 second input window.

## Requirements

Run the benchmark in a container that already contains ROCm PyTorch, torchaudio, vLLM, vLLM-Omni, and the OmniVoice runtime dependencies. The script does not create a virtual environment. It uses the `python` and `vllm` commands from the current container.

The package installation uses `--no-deps`, so it does not replace the container's PyTorch packages. The run also needs `bash`, `curl`, internet access to Hugging Face, and enough disk space for model files, audio, and results. One MI300X with 192 GB of VRAM was sufficient for the default concurrency 1 benchmark.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--small` | Off | Select 10 source passages instead of 25. |
| `--concurrency N` | Concurrency 1 only | Add a vLLM-Omni concurrency sweep. Repeat the option to add several values. |
| `-h`, `--help` | | Print usage. |

The script rejects missing, zero, negative, and repeated concurrency values.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU_INDEX` | `0` | GPU exposed through `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`. |
| `PORT` | `8091` | Port used by the temporary vLLM-Omni server. |
| `MODEL` | `k2-fsa/OmniVoice` | Hugging Face model name or local model directory. |
| `MODEL_REVISION` | `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` | OmniVoice model revision used by both backends. |
| `WHISPER_MODEL` | `openai/whisper-large-v3` | Whisper model used for transcription. |
| `WHISPER_REVISION` | `06f233fe06e710322aca913c1bc4249a0d71fce1` | Whisper model revision. |
| `WHISPER_DTYPE` | `float32` | Whisper data type. Accepted values are `float32`, `float16`, and `bfloat16`. |
| `SEEDS` | `42` | Space separated generation seeds, such as `"42 123"`. Each extra seed repeats every prompt and mode. |
| `OUTPUT_DIR` | `benchmarks/tts/omnivoice_longform/results/<timestamp>` | Result directory. |
| `BENCH_PYTHON` | `python` | Python command used by the script. |
| `VLLM_BIN` | `vllm` | vLLM command used by the script. |

Example with explicit settings:

```bash
GPU_INDEX=0 \
PORT=8095 \
SEEDS="42" \
OUTPUT_DIR=/scratch/omnivoice-results \
bash benchmarks/tts/omnivoice_longform/run_benchmark.sh --small
```

## Results

Each run creates this directory structure:

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

Read these files first:

* `evaluation/summary.md` compares coverage, word error rate, latency, RTF, failures, and peak reserved GPU memory for both backends.
* `vllm-omni/serving_summary.json` contains serving throughput and percentile metrics for every requested concurrency.
* `reference/generation.jsonl` and `vllm-omni/serving.jsonl` contain one row per measured case.
* `vllm-omni/server.log` contains server startup and request errors.
* `run_metadata.json` records the exact environment and settings needed to identify the run.

For example:

```bash
RESULT_DIR=benchmarks/tts/omnivoice_longform/results/20260819-154523
cat "$RESULT_DIR/evaluation/summary.md"
python -m json.tool "$RESULT_DIR/vllm-omni/serving_summary.json"
```

The benchmark saves vLLM-Omni audio only for concurrency 1. It keeps the JSON rows for every additional concurrency sweep.

## Resume an interrupted run

Set `OUTPUT_DIR` to the existing result directory and rerun the same command. The reference runner and Whisper evaluator save progress after every case. The vLLM-Omni runner saves progress after each completed concurrency sweep.

The benchmark refuses to resume when the model revision, Whisper revision, dependency versions, repository contents, run settings, or prompt manifest differ from the original run. Failed cases remain in the result files, so you can see which requests or transcriptions failed.

## GPU memory measurement

The benchmark records peak PyTorch reserved GPU memory for every successful generation request. The official runner reads the value inside its process. The vLLM-Omni server returns the value in the `X-Peak-Memory-MB` response header, and the client converts it to GiB.

PyTorch uses the same memory API on ROCm and CUDA. Reserved memory includes PyTorch's reusable allocator pool and does not include memory allocated outside PyTorch.
