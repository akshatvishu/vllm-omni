# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E online-serving tests for Ming-omni-tts."""

import os
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.conftest import OmniServerParams
from tests.utils import hardware_test

MODEL = "inclusionAI/Ming-omni-tts-0.5B"
STAGE_CONFIG = str(
    Path(__file__).parent.parent.parent.parent
    / "vllm_omni"
    / "model_executor"
    / "stage_configs"
    / "ming_tts_async_chunk.yaml"
)

SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=STAGE_CONFIG,
            server_args=["--enforce-eager", "--disable-log-stats"],
        ),
        id="async_chunk",
    )
]


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_ming_tts_audio_speech_non_streaming(omni_server, openai_client) -> None:
    """Test non-streaming Ming generation through /v1/audio/speech."""
    request_config = {
        "model": omni_server.model,
        "input": "我会一直在这里陪着你，直到你慢慢地沉入那个最温柔的梦里。",
        "stream": False,
        "response_format": "wav",
    }
    openai_client.send_audio_speech_request(request_config, request_num=2)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_ming_tts_audio_speech_streaming(omni_server, openai_client) -> None:
    """Test streaming Ming generation through /v1/audio/speech."""
    request_config = {
        "model": omni_server.model,
        "input": "这款产品的名字，叫变态坑爹牛肉丸。",
        "voice": "灵小甄",
        "stream": True,
        "response_format": "wav",
    }
    openai_client.send_audio_speech_request(request_config)
