#!/bin/bash
# Launch vLLM-Omni server for Ming-omni-tts.
#
# Usage:
#   ./run_server.sh
#   PORT=8000 ./run_server.sh

set -e

MODEL="${MODEL:-inclusionAI/Ming-omni-tts-0.5B}"
PORT="${PORT:-8091}"
STAGE_CONFIG="${STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/ming_tts_async_chunk.yaml}"

echo "Starting Ming-omni-tts server with model: $MODEL"
echo "Stage config: $STAGE_CONFIG"

vllm-omni serve "$MODEL" \
    --stage-configs-path "$STAGE_CONFIG" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --omni
