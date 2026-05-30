# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.model_executor.models.ming_tts.constants import (
    AGGREGATOR_HIDDEN_SIZE,
    HISTORY_PATCH_SIZE,
    LATENT_DIM,
    LLM_HIDDEN_SIZE,
    LLM_VOCAB_SIZE,
    PATCH_SIZE,
    SAMPLE_RATE,
    VAE_PATCH_SIZE,
)
from vllm_omni.model_executor.models.ming_tts.fm.cfm import Solver as MingTTSSolver
from vllm_omni.model_executor.models.ming_tts.validation import validate_ming_tts_config
from vllm_omni.model_executor.models.ming_utils.audio_vae import AudioVAEConfig
from vllm_omni.model_executor.models.ming_utils.fm import Solver

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def test_ming_tts_audio_vae_uses_common_config():
    """AudioVAEConfig is shared by Ming dense and Ming flash modules."""
    cfg = AudioVAEConfig(sample_rate=16000, patch_size=-1)

    assert cfg.sample_rate == 16000
    assert cfg.patch_size == -1


def test_ming_tts_cfm_solver_uses_common_implementation():
    """Ming dense CFM imports the shared solver implementation."""
    assert MingTTSSolver is Solver


def test_ming_dense_validation_rejects_semantic_audio_vae_config():
    """Dense 0.5B validation rejects semantic AudioVAE configs."""
    cfg = SimpleNamespace(
        audio_dummy_token_id=151705,
        audio_eos_token_id=151704,
        text_eos_token_id=151669,
        audio_tokenizer_config=AudioVAEConfig(
            sample_rate=SAMPLE_RATE,
            patch_size=VAE_PATCH_SIZE,
            semantic_module_kwargs={"whisper_encoder": {}},
            enc_kwargs={"latent_dim": LATENT_DIM, "input_dim": 882, "hop_size": 882},
            dec_kwargs={"latent_dim": LATENT_DIM, "output_dim": 882},
        ),
        latent_dim=LATENT_DIM,
        patch_size=PATCH_SIZE,
        history_patch_size=HISTORY_PATCH_SIZE,
        llm_hidden_size=LLM_HIDDEN_SIZE,
        llm_vocab_size=LLM_VOCAB_SIZE,
        sample_rate=SAMPLE_RATE,
        vae_patch_size=VAE_PATCH_SIZE,
        llm_config={"hidden_size": LLM_HIDDEN_SIZE},
        aggregator_config={"hidden_size": AGGREGATOR_HIDDEN_SIZE},
        ditar_config={"hidden_size": AGGREGATOR_HIDDEN_SIZE},
        latent_chunk_size=1,
        latent_left_context=0,
        max_decode_steps=1,
        stop_head_threshold=0.5,
        stop_head_min_steps=0,
    )

    with pytest.raises(ValueError, match="semantic_module_kwargs"):
        validate_ming_tts_config(cfg)


def test_ming_instruction_parser_preserves_dense_and_flash_defaults():
    """Ming dense and Ming flash keep distinct instruction defaults."""
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving.uploaded_speakers = {"uploaded": {}}

    dense_plain = serving._parse_ming_instruction(SimpleNamespace(instructions="calm", language=None, voice=None))
    assert dense_plain == "calm"

    dense_with_fields = serving._parse_ming_instruction(
        SimpleNamespace(instructions="calm", language="Auto", voice="灵小甄")
    )
    assert dense_with_fields == {"IP": "灵小甄", "风格": "calm"}

    flash_fields = serving._parse_ming_instruction_fields(
        SimpleNamespace(instructions="calm", language="粤语", voice="灵小甄")
    )
    assert flash_fields == {"风格": "calm"}
