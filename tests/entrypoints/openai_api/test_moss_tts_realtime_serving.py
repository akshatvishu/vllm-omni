# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.openai import serving_speech as serving_speech_module
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_realtime_components_use_the_realtime_model_and_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    server = object.__new__(OmniOpenAIServingSpeech)
    server.engine_client = SimpleNamespace(model_config=SimpleNamespace(model="OpenMOSS-Team/MOSS-TTS-Realtime"))

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
    assert codec_calls == [("OpenMOSS-Team/MOSS-Audio-Tokenizer", True)]
    assert processor_calls == [tokenizer]


def test_realtime_serving_builds_the_talker_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    server = object.__new__(OmniOpenAIServingSpeech)
    server._moss_variant = "realtime"
    server._speaker_cache = object()
    tokenizer = object()
    processor = object()
    codec = object()
    server._get_moss_realtime_components = lambda: (tokenizer, processor, codec)
    server.uploaded_speakers = {}

    async def resolve_ref_audio(ref_audio):
        assert ref_audio == "data:audio/wav;base64,AAAA"
        return [0.1, 0.2], 24000, "resolved-audio-key"

    server._resolve_ref_audio = resolve_ref_audio

    reference_codes = torch.arange(64, dtype=torch.int64).reshape(4, 16)
    encode_call = None

    async def fake_encode(ref_audio, **kwargs):
        nonlocal encode_call
        resolved_audio = await kwargs.pop("resolve_ref_audio")(ref_audio)
        encode_call = (ref_audio, kwargs, resolved_audio)
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

    assert encode_call == (
        "data:audio/wav;base64,AAAA",
        {
            "codec": codec,
            "speaker_cache": server._speaker_cache,
            "voice_name": None,
            "voice_created_at": 0,
        },
        ([0.1, 0.2], 24000, "resolved-audio-key"),
    )
    assert build_call == (tokenizer, processor, "speak this text", reference_codes)
    assert params == {
        "prompt_token_ids": [10, 11],
        "codes": {"ref": reference_codes},
        "ids": {"all": [12]},
        "max_new_frames": [50],
        "ref_audio_cache_key": "resolved-audio-key",
    }
    assert "prompt_audio_array" not in params


@pytest.mark.asyncio
async def test_realtime_reference_cache_tracks_resolved_audio_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.model_executor.models.moss_tts.reference_encoder import (
        encode_realtime_reference_codes,
    )
    from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

    class Codec:
        def __init__(self):
            self.call_count = 0

        def batch_encode(self, wavs, *, num_quantizers):
            assert num_quantizers == 16
            self.call_count += 1
            return SimpleNamespace(
                audio_codes=torch.full((16, 1, 3), self.call_count, dtype=torch.int64),
                audio_codes_lengths=torch.tensor([3]),
            )

    resolve_keys = iter(("key-a", "key-b"))

    async def resolve_ref_audio(_):
        return [0.1, 0.2, 0.3], 24000, next(resolve_keys)

    async def run_inline(func, *args):
        return func(*args)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    codec = Codec()
    speaker_cache = SpeakerEmbeddingCache(max_bytes=1024 * 1024)
    first = await encode_realtime_reference_codes(
        "file:///speaker.wav",
        codec=codec,
        resolve_ref_audio=resolve_ref_audio,
        speaker_cache=speaker_cache,
    )
    second = await encode_realtime_reference_codes(
        "file:///speaker.wav",
        codec=codec,
        resolve_ref_audio=resolve_ref_audio,
        speaker_cache=speaker_cache,
    )

    assert codec.call_count == 2
    assert torch.all(first == 1)
    assert torch.all(second == 2)
