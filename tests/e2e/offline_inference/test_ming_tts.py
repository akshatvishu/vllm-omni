# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end offline inference tests for Ming-omni-tts."""

import asyncio
import os
import uuid
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from tests.utils import hardware_test
from vllm_omni import AsyncOmni, Omni
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import KEY_MAX_DECODE_STEPS, TEXT_EOS_TOKEN_ID
from vllm_omni.model_executor.models.ming_tts.prompt_builder import build_ming_dense_prompt

MODEL = "inclusionAI/Ming-omni-tts-0.5B"
STAGE_CONFIG = str(
    Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / "ming_tts.yaml"
)
STREAM_STAGE_CONFIG = str(
    Path(__file__).parent.parent.parent.parent
    / "vllm_omni"
    / "model_executor"
    / "stage_configs"
    / "ming_tts_async_chunk.yaml"
)
TEST_TEXT = "我会一直在这里陪着你，直到你慢慢地沉入那个最温柔的梦里。"
TEST_INSTRUCTION = "轻柔的ASMR耳语，慢速，贴近麦克风"
MIN_AUDIO_SAMPLES = 1000


def _build_prompt() -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
    return build_ming_dense_prompt(
        tokenizer,
        prompt="Please generate speech based on the following description.\n",
        text=TEST_TEXT,
        instruction=TEST_INSTRUCTION,
        runtime_controls={KEY_MAX_DECODE_STEPS: 200},
        use_zero_spk_emb=True,
    )


def _sampling_params_list() -> list[SamplingParams]:
    return [
        SamplingParams(
            temperature=0.0,
            max_tokens=201,
            stop_token_ids=[int(TEXT_EOS_TOKEN_ID)],
        ),
        SamplingParams(temperature=0.0, max_tokens=1),
    ]


def _flatten_audio(audio) -> torch.Tensor:
    if isinstance(audio, list):
        parts = [torch.as_tensor(item, dtype=torch.float32).reshape(-1).cpu() for item in audio]
        parts = [item for item in parts if item.numel() > 0]
        if not parts:
            return torch.zeros((0,), dtype=torch.float32)
        return torch.cat(parts, dim=0)
    return torch.as_tensor(audio, dtype=torch.float32).reshape(-1).cpu()


def _extract_audio(multimodal_output: dict) -> torch.Tensor:
    audio = multimodal_output.get("audio")
    if audio is None:
        raise RuntimeError("Expected multimodal_output['audio']")
    waveform = _flatten_audio(audio)
    if waveform.numel() == 0:
        raise RuntimeError("Generated audio waveform is empty")
    return waveform


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ming_tts_offline_basic() -> None:
    """Test blocking Ming generation through Omni."""
    omni = Omni(
        model=MODEL,
        stage_configs_path=STAGE_CONFIG,
        stage_init_timeout=300,
        enforce_eager=True,
    )
    try:
        outputs = omni.generate(
            prompts=[_build_prompt()],
            sampling_params_list=_sampling_params_list(),
            py_generator=False,
        )
        final_output = next((item for item in outputs if item.final_output_type == "audio"), None)
        assert final_output is not None, "No final audio output produced"
        waveform = _extract_audio(final_output.multimodal_output or {})
        assert waveform.numel() > MIN_AUDIO_SAMPLES
        assert np.max(np.abs(waveform.numpy())) > 0.01, "Audio appears silent"
    finally:
        omni.close()


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ming_tts_offline_streaming() -> None:
    """Test async_chunk streaming Ming generation through AsyncOmni."""

    async def _run() -> None:
        async_omni = AsyncOmni(
            model=MODEL,
            stage_configs_path=STREAM_STAGE_CONFIG,
            stage_init_timeout=300,
            enforce_eager=True,
        )
        try:
            all_audio_chunks = []
            accumulated_samples = 0
            chunk_idx = 0
            async for stage_output in async_omni.generate(
                prompt=_build_prompt(),
                request_id=str(uuid.uuid4()),
                sampling_params_list=_sampling_params_list(),
            ):
                multimodal_output = stage_output.multimodal_output or {}
                audio = multimodal_output.get("audio")
                if audio is None:
                    continue
                finished = stage_output.finished
                if isinstance(audio, torch.Tensor):
                    if finished:
                        audio_chunk = audio[accumulated_samples:].float().detach().cpu()
                    else:
                        audio_chunk = audio.float().detach().cpu()
                elif isinstance(audio, list):
                    audio_chunk = torch.as_tensor(audio[chunk_idx], dtype=torch.float32).reshape(-1).cpu()
                else:
                    audio_chunk = torch.as_tensor(audio, dtype=torch.float32).reshape(-1).cpu()
                accumulated_samples += int(audio_chunk.numel())
                chunk_idx += 1
                if audio_chunk.numel() > 0:
                    all_audio_chunks.append(audio_chunk)
            assert all_audio_chunks, "No streaming audio chunks received"
            waveform = torch.cat(all_audio_chunks, dim=0)
            assert waveform.numel() > MIN_AUDIO_SAMPLES
            assert np.max(np.abs(waveform.numpy())) > 0.01, "Audio appears silent"
        finally:
            async_omni.shutdown()

    asyncio.run(_run())
