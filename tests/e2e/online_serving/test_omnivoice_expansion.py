# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for OmniVoice TTS model via /v1/audio/speech endpoint.

Tests verify that the OmniVoice model generates valid audio when
accessed through the standard OpenAI-compatible speech API.
"""

import os
from io import BytesIO

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

try:
    from transformers import HiggsAudioV2TokenizerModel  # noqa: F401

    _HAS_VOICE_CLONE = True
except ImportError:
    _HAS_VOICE_CLONE = False

pytestmark = [pytest.mark.slow, pytest.mark.tts]

MODEL = "k2-fsa/OmniVoice"

STAGE_CONFIG = get_deploy_config_path("omnivoice.yaml")
EXTRA_ARGS = [
    "--trust-remote-code",
    "--disable-log-stats",
]
TEST_PARAMS = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=STAGE_CONFIG,
        server_args=EXTRA_ARGS,
    )
]

# Lower this in ``request_config`` via ``min_audio_bytes`` if a run produces legitimately short WAVs.
_DEFAULT_MIN_AUDIO_BYTES = 5000
_OMNIVOICE_REF_AUDIO_SEED = 102


REF_AUDIO_URL = load_test_audio_data_url("qwen3_tts/clone_2.wav")
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


def get_prompt(prompt_type="text"):
    prompts = {
        "text": "The weather is nice today, perfect for a walk in the park.",
    }
    return prompts.get(prompt_type, prompts["text"])


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestOmniVoiceTTS:
    """E2E tests for OmniVoice TTS model."""

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speech_auto_voice(self, omni_server, openai_client) -> None:
        """Test auto voice TTS generation (text only, no reference audio)."""
        request_config = {
            "model": omni_server.model,
            "input": get_prompt("text"),
            "response_format": "wav",
            "timeout": 180.0,
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }
        openai_client.send_audio_speech_request(request_config)


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestOmniVoiceSeed:
    """E2E tests for OmniVoice Seed Params."""

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speech_auto_voice_seed_deterministic(self, omni_server, openai_client) -> None:
        cfg = {
            "model": omni_server.model,
            "input": get_prompt("text"),
            "response_format": "wav",
            "seed": 42,
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }

        r1 = openai_client.send_audio_speech_request(cfg)[0]
        r2 = openai_client.send_audio_speech_request(cfg)[0]
        assert r1.audio_bytes is not None
        assert r2.audio_bytes is not None
        assert r1.audio_bytes == r2.audio_bytes

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speech_auto_voice_seed_non_deterministic(self, omni_server, openai_client) -> None:
        cfg = {
            "model": omni_server.model,
            "input": get_prompt("text"),
            "response_format": "wav",
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }

        cfg1 = {**cfg, "seed": 42}
        cfg2 = {**cfg, "seed": 43}

        r1 = openai_client.send_audio_speech_request(cfg1)[0]
        r2 = openai_client.send_audio_speech_request(cfg2)[0]
        assert r1.audio_bytes != r2.audio_bytes


@pytest.mark.skipif(not _HAS_VOICE_CLONE, reason="Voice cloning requires transformers>=5.3.0")
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestOmniVoiceVoiceCloning:
    """E2E tests for OmniVoice voice cloning functionality."""

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_ref_audio_only(self, omni_server, openai_client) -> None:
        """Test automatic reference transcription with ref_audio only."""
        request_config = {
            "model": omni_server.model,
            "input": "hello",
            "ref_audio": REF_AUDIO_URL,
            "response_format": "wav",
            "seed": _OMNIVOICE_REF_AUDIO_SEED,
            "timeout": 180.0,
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
            "transcript_language": "en",
        }
        response = openai_client.send_audio_speech_request(request_config)[0]
        assert response.audio_bytes is not None
        audio, sample_rate = sf.read(BytesIO(response.audio_bytes), dtype="float32")
        assert sample_rate == 24000
        assert np.isfinite(audio).all()
        assert np.unique(audio).size > 1
        assert np.sqrt(np.mean(audio**2)) > 0.01

        repeated_response = openai_client.send_audio_speech_request(request_config)[0]
        assert repeated_response.audio_bytes is not None
        repeated_audio, repeated_sample_rate = sf.read(BytesIO(repeated_response.audio_bytes), dtype="float32")
        assert repeated_sample_rate == sample_rate
        assert repeated_audio.shape == audio.shape
        np.testing.assert_allclose(repeated_audio, audio, rtol=1e-5, atol=1e-4)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_ref_audio_and_text(self, omni_server, openai_client) -> None:
        """Test voice cloning with ref_audio and ref_text (in-context mode)."""
        request_config = {
            "model": omni_server.model,
            "input": get_prompt("text"),
            "ref_audio": REF_AUDIO_URL,
            "ref_text": REF_TEXT,
            "response_format": "wav",
            "timeout": 180.0,
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }
        openai_client.send_audio_speech_request(request_config)
