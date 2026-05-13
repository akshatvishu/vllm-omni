# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E online-serving tests for Ming-omni-tts."""

import concurrent.futures
import io
import os
import wave

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import SAMPLE_RATE

MODEL = "inclusionAI/Ming-omni-tts-0.5B"
DEPLOY_CONFIG = get_deploy_config_path("ming_tts.yaml")

SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            server_args=["--enforce-eager", "--disable-log-stats"],
        ),
        id="async_chunk",
    )
]


def _wav_sample_rate(audio_bytes: bytes) -> int:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        return int(wav_file.getframerate())


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
    request_inputs = [
        "我会一直在这里陪着你，直到你慢慢地沉入那个最温柔的梦里。",
        "这款产品的名字，叫变态坑爹牛肉丸。",
    ]

    def _send_one(text):
        per_request_config = {**request_config, "input": text}
        responses = openai_client.send_audio_speech_request(per_request_config)
        assert len(responses) == 1
        return text, responses[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request_inputs)) as executor:
        futures = [executor.submit(_send_one, text) for text in request_inputs]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    assert {text for text, _ in results} == set(request_inputs)
    assert len(results) == len(request_inputs)
    for _, response in results:
        assert response.audio_bytes is not None, "Expected WAV bytes from /v1/audio/speech"
        sample_rate = _wav_sample_rate(response.audio_bytes)
        assert sample_rate == SAMPLE_RATE, f"Expected Ming output sample rate {SAMPLE_RATE}, got {sample_rate}"


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
    responses = openai_client.send_audio_speech_request(request_config)
    assert len(responses) == 1
    assert responses[0].audio_bytes is not None, "Expected streamed WAV bytes from /v1/audio/speech"
    sample_rate = _wav_sample_rate(responses[0].audio_bytes)
    assert sample_rate == SAMPLE_RATE, f"Expected Ming output sample rate {SAMPLE_RATE}, got {sample_rate}"
