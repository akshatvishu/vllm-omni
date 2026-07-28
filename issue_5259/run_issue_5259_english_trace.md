# Run issue 5259 in English

Run every command inside the ROCm Docker container.

## Check the code

```bash
cd /scratch/workspace/vllm-omni

git rev-parse --short HEAD
python -c 'import vllm_omni; print(vllm_omni.__file__)'
```

The imported file must be under `/scratch/workspace/vllm-omni`.

If Python imports `/app/vllm-omni`, install the current checkout and check again:

```bash
uv pip install --system -e /scratch/workspace/vllm-omni
hash -r
python -c 'import vllm_omni; print(vllm_omni.__file__)'
```

Stop any old vLLM server before starting the new server.

## Start the server

Run this command in the first SSH shell:

```bash
cd /scratch/workspace/vllm-omni
set -o pipefail

TRACE_DIR="$PWD/issue_5259_server_trace_eos"
mkdir -p "$TRACE_DIR"

MINICPMO45_TRACE=1 \
MINICPMO45_TRACE_DIR="$TRACE_DIR" \
TMPDIR="$TRACE_DIR" \
vllm serve openbmb/MiniCPM-o-4_5 \
  --omni \
  --port 28889 \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --interleave-mm-strings \
  2>&1 | tee "$TRACE_DIR/server.log"
```

Wait until the server is ready before sending the request.

## Send one request

Run this command in the second SSH shell:

```bash
cd /scratch/workspace/vllm-omni
set -o pipefail

REPO_DIR="$PWD"
TRACE_DIR="$REPO_DIR/issue_5259_server_trace_eos"

(
  cd "$TRACE_DIR"
  python "$REPO_DIR/examples/online_serving/minicpmo/openai_chat_completion_client_for_multimodal_generation.py" \
    --model openbmb/MiniCPM-o-4_5 \
    --query-type text \
    --modalities text,audio \
    --port 28889 \
    --prompt "Hello, could you tell me a 500-character story?" \
    2>&1 | tee client.log
)
```

You will find `server.log`, `client.log`, and the generated WAV file in the trace folder.

## Collect the trace

Run this command after the request finishes:

```bash
cd /scratch/workspace/vllm-omni

TRACE_DIR="$PWD/issue_5259_server_trace_eos"

rg --no-filename '\[MiniCPMO45Trace\]' \
  "$TRACE_DIR" \
  --glob '*.log' \
  > "$TRACE_DIR/trace.log"

tar -czf "$PWD/issue_5259_server_trace_eos.tar.gz" \
  -C "$TRACE_DIR" .
```

The folder also contains exact `talker_condition`, `talker_sampling`, and
`talker_eos` tensor artifacts. Do not send another request if `trace.log`
contains `talker_eos_sampling` and its `artifact.error` value is `null`.
