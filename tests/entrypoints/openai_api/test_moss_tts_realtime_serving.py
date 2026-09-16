# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from vllm_omni.entrypoints.openai import serving_speech as serving_speech_module
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.model_executor.models.moss_tts import reference_encoder
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("hf_config", "expected_codec_path"),
    [
        (SimpleNamespace(), "OpenMOSS-Team/MOSS-Audio-Tokenizer"),
        (
            SimpleNamespace(codec_model_name_or_path="/models/custom-moss-codec"),
            "/models/custom-moss-codec",
        ),
    ],
    ids=["default-codec", "configured-codec"],
)
def test_realtime_components_use_the_realtime_model_and_codec(
    monkeypatch: pytest.MonkeyPatch,
    hf_config: SimpleNamespace,
    expected_codec_path: str,
) -> None:
    server = object.__new__(OmniOpenAIServingSpeech)
    server.engine_client = SimpleNamespace(
        model_config=SimpleNamespace(
            model="OpenMOSS-Team/MOSS-TTS-Realtime",
            hf_config=hf_config,
        )
    )

    tokenizer = object()
    codec = type("Codec", (), {"to": lambda self, device: self, "eval": lambda self: self})()
    processor_calls = []

    class Processor:
        def __init__(self, *, tokenizer):
            processor_calls.append(tokenizer)

    class_calls = []
    tokenizer_calls = []
    codec_calls = []

    def fake_get_class(class_reference, model_id):
        class_calls.append((class_reference, model_id))
        return Processor

    def fake_load_tokenizer(model_id, *, trust_remote_code):
        tokenizer_calls.append((model_id, trust_remote_code))
        return tokenizer

    def fake_load_codec(model_id, *, trust_remote_code):
        codec_calls.append((model_id, trust_remote_code))
        return codec

    monkeypatch.setattr(serving_speech_module, "get_class_from_dynamic_module", fake_get_class)
    monkeypatch.setattr(serving_speech_module.AutoTokenizer, "from_pretrained", fake_load_tokenizer)
    monkeypatch.setattr(serving_speech_module.AutoModel, "from_pretrained", fake_load_codec)

    components = server._get_moss_realtime_components()

    assert components[0] is tokenizer
    assert components[2] is codec
    assert server._get_moss_realtime_components() is components
    assert class_calls == [
        (
            "processing_mossttsrealtime.MossTTSRealtimeProcessor",
            "OpenMOSS-Team/MOSS-TTS-Realtime",
        )
    ]
    assert tokenizer_calls == [("OpenMOSS-Team/MOSS-TTS-Realtime", True)]
    assert codec_calls == [(expected_codec_path, True)]
    assert processor_calls == [tokenizer]


def test_realtime_serving_builds_the_talker_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    server = object.__new__(OmniOpenAIServingSpeech)
    server._moss_variant = "realtime"
    server._moss_realtime_components_lock = asyncio.Lock()
    server._speaker_cache = object()
    tokenizer = object()
    processor = object()
    codec = object()
    request_thread = threading.get_ident()

    def load_components():
        assert server._moss_realtime_components_lock.locked()
        assert threading.get_ident() != request_thread
        return tokenizer, processor, codec

    server._get_moss_realtime_components = load_components

    async def fake_resolve(ref_audio):
        assert ref_audio == "data:audio/wav;base64,AAAA"
        return [[0.0]], 24_000, "reference-cache-key"

    server._resolve_ref_audio = fake_resolve

    reference_codes = torch.arange(64, dtype=torch.int64).reshape(4, 16)
    encode_call = None

    async def fake_encode(ref_audio, **kwargs):
        nonlocal encode_call
        encode_call = (ref_audio, kwargs)
        assert await kwargs["resolve_ref_audio"](ref_audio) == ([[0.0]], 24_000, "reference-cache-key")
        return reference_codes

    build_call = None

    def fake_build(actual_tokenizer, actual_processor, text, actual_codes):
        nonlocal build_call
        build_call = (actual_tokenizer, actual_processor, text, actual_codes)
        return {
            "prompt_token_ids": [10, 11],
            "codes": {"ref": actual_codes},
            "ids": {"all": [12]},
        }

    monkeypatch.setattr(serving_speech_module, "encode_realtime_reference_codes", fake_encode)
    monkeypatch.setattr(serving_speech_module, "build_realtime_prompt", fake_build)

    params = asyncio.run(
        server._build_moss_tts_params(
            OpenAICreateSpeechRequest(
                input="speak this text",
                ref_audio="data:audio/wav;base64,AAAA",
                max_new_tokens=50,
            )
        )
    )

    assert encode_call is not None
    assert encode_call[0] == "data:audio/wav;base64,AAAA"
    assert encode_call[1]["codec"] is codec
    assert encode_call[1]["speaker_cache"] is server._speaker_cache
    assert encode_call[1]["voice_name"] is None
    assert encode_call[1]["voice_created_at"] == 0
    assert build_call == (tokenizer, processor, "speak this text", reference_codes)
    assert params == {
        "prompt_token_ids": [10, 11],
        "codes": {"ref": reference_codes},
        "ids": {"all": [12]},
        "max_new_frames": [50],
        "ref_audio_cache_key": "reference-cache-key",
    }
    assert "prompt_audio_array" not in params


@pytest.mark.parametrize("realtime", [False, True])
@pytest.mark.parametrize("voice_name", [None, "speaker"])
def test_reference_cache_reuses_codes(monkeypatch, realtime, voice_name):
    cache = SpeakerEmbeddingCache()
    resolve = AsyncMock(return_value=([0.0], 24000, "audio-key"))
    request_thread = threading.get_ident()
    encode_calls = []
    expected = torch.arange(64).reshape(4, 16)

    def encode(*args, **kwargs):
        assert threading.get_ident() != request_thread
        encode_calls.append((args, kwargs))
        return expected.clone()

    if realtime:
        monkeypatch.setattr(reference_encoder, "_encode_realtime_wav_sync", encode)
        encode_reference = reference_encoder.encode_realtime_reference_codes
        encoder_kwargs = {"codec": object()}
    else:
        monkeypatch.setattr(reference_encoder, "_encode_wav_sync", encode)
        encode_reference = reference_encoder.encode_reference_codes
        encoder_kwargs = {"processor": object(), "variant": "tts", "n_vq": 16, "sr_target": 24000}

    async def run():
        kwargs = dict(
            **encoder_kwargs,
            resolve_ref_audio=resolve,
            speaker_cache=cache,
            voice_name=voice_name,
            voice_created_at=1,
        )
        first = await encode_reference("reference", **kwargs)
        second = await encode_reference("reference", **kwargs)
        assert torch.equal(first, expected)
        assert torch.equal(second, expected)
        second.zero_()
        third = await encode_reference("reference", **kwargs)
        assert torch.equal(third, expected)
        assert len(encode_calls) == 1
        assert resolve.await_count == (1 if voice_name else 3)
        model_type = "moss_tts_realtime_nq16" if realtime else "moss_tts_tts_nq16"
        key = cache.make_cache_key(voice_name or "ref:audio-key", model_type, 1 if voice_name else 0)
        assert torch.equal(cache.get(key)["codes"], expected)
        if voice_name:
            kwargs["voice_created_at"] = 2
            await encode_reference("reference", **kwargs)
            assert len(encode_calls) == 2
            assert resolve.await_count == 2

    asyncio.run(run())


@pytest.mark.parametrize("realtime", [False, True])
def test_failed_reference_encoding_does_not_populate_cache(monkeypatch, realtime):
    cache = SpeakerEmbeddingCache()
    resolve = AsyncMock(return_value=([0.0], 24000, "audio-key"))

    def fail(*args, **kwargs):
        raise ValueError("invalid reference")

    if realtime:
        monkeypatch.setattr(reference_encoder, "_encode_realtime_wav_sync", fail)
        encode_reference = reference_encoder.encode_realtime_reference_codes
        kwargs = {"codec": object()}
    else:
        monkeypatch.setattr(reference_encoder, "_encode_wav_sync", fail)
        encode_reference = reference_encoder.encode_reference_codes
        kwargs = {"processor": object(), "variant": "tts", "n_vq": 16, "sr_target": 24000}

    with pytest.raises(ValueError, match="invalid reference"):
        asyncio.run(encode_reference("reference", resolve_ref_audio=resolve, speaker_cache=cache, **kwargs))
    assert cache.memory_bytes() == 0
