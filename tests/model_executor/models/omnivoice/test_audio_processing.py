# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from vllm_omni.diffusion.models.omnivoice import audio as audio_utils
from vllm_omni.diffusion.models.omnivoice import pipeline_omnivoice
from vllm_omni.diffusion.models.omnivoice.audio import (
    PreparedReferenceAudio,
    add_reference_punctuation,
    postprocess_generated_audio,
    prepare_reference_audio,
    remove_silence,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SAMPLE_RATE = 24000
_HOP_LENGTH = 960


def _prepare(waveform, sample_rate=_SAMPLE_RATE):
    return prepare_reference_audio(
        waveform,
        sample_rate,
        target_sample_rate=_SAMPLE_RATE,
        hop_length=_HOP_LENGTH,
        trim_long=False,
    )


def test_reference_audio_normalization_and_hop_alignment():
    waveform = np.full(1010, 0.05, dtype=np.float32)

    prepared = _prepare(waveform)

    assert prepared.waveform.shape == (1, _HOP_LENGTH)
    assert prepared.original_rms == pytest.approx(0.05, abs=1e-6)
    assert np.sqrt(np.mean(prepared.waveform**2)) == pytest.approx(0.1, abs=1e-4)


def test_reference_audio_at_or_above_target_rms_is_not_rescaled():
    prepared = _prepare(np.full(960, 0.2, dtype=np.float32))

    assert prepared.original_rms == pytest.approx(0.2, abs=1e-6)
    assert np.sqrt(np.mean(prepared.waveform**2)) == pytest.approx(0.2, abs=1e-4)


def test_reference_audio_resamples_and_downmixes():
    waveform = np.full((2, 800), 0.1, dtype=np.float32)

    prepared = _prepare(waveform, sample_rate=16000)

    assert prepared.waveform.shape == (1, 960)
    assert prepared.sample_rate == _SAMPLE_RATE


@pytest.mark.parametrize("waveform", [np.full(960, 0.1, dtype=np.float32), torch.full((960,), 0.1)])
def test_reference_audio_does_not_mutate_input(waveform):
    original = waveform.clone() if isinstance(waveform, torch.Tensor) else waveform.copy()

    _prepare(waveform)

    if isinstance(waveform, torch.Tensor):
        assert torch.equal(waveform, original)
    else:
        np.testing.assert_array_equal(waveform, original)


def test_reference_audio_rejects_empty_after_silence_removal():
    with pytest.raises(ValueError, match="empty after silence removal"):
        _prepare(np.zeros(960, dtype=np.float32))


def test_reference_audio_matches_official_asset_preparation():
    asset_path = Path(__file__).resolve().parents[4] / "tests/assets/qwen3_tts/clone_2.wav"
    waveform, sample_rate = sf.read(asset_path, always_2d=False)

    prepared = prepare_reference_audio(
        waveform,
        sample_rate,
        target_sample_rate=_SAMPLE_RATE,
        hop_length=_HOP_LENGTH,
        trim_long=True,
    )

    assert prepared.waveform.shape == (1, 179520)
    assert prepared.original_rms == pytest.approx(0.07468347, abs=1e-5)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello", "hello."),
        ("你好", "你好。"),
        ("hello!", "hello!"),
        ("hello,", "hello,"),
        ("hello]", "hello]"),
        ("", ""),
    ],
)
def test_reference_punctuation(text, expected):
    assert add_reference_punctuation(text) == expected


def test_remove_silence_uses_reference_thresholds():
    waveform = np.concatenate(
        [
            np.zeros(2400, dtype=np.float32),
            np.full(12000, 0.2, dtype=np.float32),
            np.zeros(9600, dtype=np.float32),
            np.full(12000, 0.2, dtype=np.float32),
            np.zeros(7200, dtype=np.float32),
        ]
    )[np.newaxis, :]

    processed = remove_silence(
        waveform,
        _SAMPLE_RATE,
        middle_silence_ms=200,
        leading_silence_ms=100,
        trailing_silence_ms=200,
    )

    assert processed.shape[-1] < waveform.shape[-1]
    assert processed.shape[-1] > 24000


def test_generated_audio_scales_with_reference_rms_and_pads():
    audio = np.full((1, 10000), 0.2, dtype=np.float32)

    processed = postprocess_generated_audio(
        audio,
        sample_rate=_SAMPLE_RATE,
        reference_rms=0.05,
    )

    # pydub rounds 416.666 ms to 417 ms before converting back to samples.
    assert processed.shape[-1] == 14808
    assert processed[0, 5000] == pytest.approx(0.1, abs=1e-5)
    assert processed[0, 0] == 0.0
    assert processed[0, -1] == 0.0


def test_generated_audio_keeps_reference_rms_at_or_above_target():
    processed = postprocess_generated_audio(
        np.full((1, 10000), 0.2, dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
        reference_rms=0.2,
    )

    assert processed[0, 5000] == pytest.approx(0.2, abs=3e-5)


def test_generated_text_only_audio_peak_normalizes():
    audio = np.full((1, 10000), 0.2, dtype=np.float32)

    processed = postprocess_generated_audio(
        audio,
        sample_rate=_SAMPLE_RATE,
        reference_rms=None,
    )

    assert processed[0, 5000] == pytest.approx(0.5, abs=1e-5)


def test_generated_audio_uses_output_silence_threshold(monkeypatch):
    calls = []

    def fake_remove_silence(audio, sample_rate, **kwargs):
        calls.append((sample_rate, kwargs))
        return audio

    monkeypatch.setattr(audio_utils, "remove_silence", fake_remove_silence)
    audio_utils.postprocess_generated_audio(
        np.ones((1, 1000), dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
        reference_rms=None,
    )

    assert calls == [
        (
            _SAMPLE_RATE,
            {
                "middle_silence_ms": 500,
                "leading_silence_ms": 100,
                "trailing_silence_ms": 100,
            },
        )
    ]


def test_generated_zero_audio_remains_finite():
    processed = postprocess_generated_audio(
        np.zeros((1, 1000), dtype=np.float32),
        sample_rate=_SAMPLE_RATE,
        reference_rms=None,
    )

    assert np.isfinite(processed).all()


class _FakeASR:
    def __init__(self, text: str):
        self.text = text
        self.inputs = []

    def __call__(self, audio_input):
        self.inputs.append(audio_input)
        return {"text": self.text}


class _FakeGenerator:
    def __init__(self, num_codebooks: int):
        self.num_codebooks = num_codebooks

    def __call__(self, **kwargs):
        target_len = kwargs["target_lens"][0]
        return torch.zeros((1, self.num_codebooks, target_len), dtype=torch.long)


class _FakeDecoder:
    def __call__(self, tokens):
        return torch.full((1, 1, 10000), 0.2, dtype=torch.float32)


class _FakeDurationEstimator:
    def __init__(self):
        self.calls = []

    def estimate_duration(self, text, ref_text, ref_audio_tokens):
        self.calls.append((text, ref_text, ref_audio_tokens))
        return 4


def _build_fake_pipeline(monkeypatch, prepared_waveform, *, asr_text="reference transcript"):
    model = pipeline_omnivoice.OmniVoicePipeline.__new__(pipeline_omnivoice.OmniVoicePipeline)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.config = SimpleNamespace(num_audio_codebook=2, audio_mask_id=-1)
    model.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(sample_rate=_SAMPLE_RATE, hop_length=_HOP_LENGTH))
    model.tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[1, 2]))
    model.generator = _FakeGenerator(num_codebooks=2)
    model.decoder = _FakeDecoder()
    model.duration_estimator = _FakeDurationEstimator()
    model._asr_pipeline = _FakeASR(asr_text)
    model.num_step = 1
    model.guidance_scale = 1.0
    model.t_shift = 1.0
    model.layer_penalty_factor = 0.0
    model.position_temperature = 1.0
    model.class_temperature = 1.0
    model.sample_rate = _SAMPLE_RATE

    prepare_calls = []
    encoded_calls = []

    def fake_prepare(waveform, sample_rate, **kwargs):
        prepare_calls.append((waveform, sample_rate, kwargs))
        return PreparedReferenceAudio(prepared_waveform, _SAMPLE_RATE, 0.07)

    def fake_encode(self, audio_signal, sample_rate):
        encoded_calls.append((audio_signal.detach().clone(), sample_rate))
        return torch.zeros((2, 3), dtype=torch.long)

    monkeypatch.setattr(pipeline_omnivoice, "prepare_reference_audio", fake_prepare)
    monkeypatch.setattr(pipeline_omnivoice.OmniVoicePipeline, "_encode_ref_audio", fake_encode)
    return model, prepare_calls, encoded_calls


def _request(prompt):
    return SimpleNamespace(
        prompts=[prompt],
        sampling_params=SimpleNamespace(extra_args={}),
    )


def test_pipeline_uses_one_prepared_waveform_for_asr_and_tokenizer(monkeypatch):
    prepared = np.arange(_HOP_LENGTH, dtype=np.float32)[np.newaxis, :]
    model, prepare_calls, encoded_calls = _build_fake_pipeline(monkeypatch, prepared)

    result = model.forward(
        _request(
            {
                "prompt": "hello",
                "multi_modal_data": {"audio": (np.ones(1000, dtype=np.float32), 16000)},
            }
        )
    )

    assert result.error is None
    assert len(prepare_calls) == 1
    assert prepare_calls[0][1] == 16000
    assert prepare_calls[0][2]["trim_long"] is True
    assert prepare_calls[0][2]["target_sample_rate"] == _SAMPLE_RATE
    assert prepare_calls[0][2]["hop_length"] == _HOP_LENGTH
    assert len(model._asr_pipeline.inputs) == 1
    np.testing.assert_array_equal(model._asr_pipeline.inputs[0]["array"], prepared[0])
    assert model._asr_pipeline.inputs[0]["sampling_rate"] == _SAMPLE_RATE
    assert len(encoded_calls) == 1
    torch.testing.assert_close(encoded_calls[0][0], torch.from_numpy(prepared))
    assert encoded_calls[0][1] == _SAMPLE_RATE
    assert model.duration_estimator.calls == [("hello", "reference transcript.", 3)]


def test_pipeline_explicit_reference_text_prepares_audio_without_asr(monkeypatch):
    prepared = np.ones((1, _HOP_LENGTH), dtype=np.float32)
    model, prepare_calls, encoded_calls = _build_fake_pipeline(monkeypatch, prepared)

    result = model.forward(
        _request(
            {
                "prompt": "hello",
                "ref_audio": (np.ones(1000, dtype=np.float32), 16000),
                "ref_text": "caller supplied transcript",
            }
        )
    )

    assert result.error is None
    assert model._asr_pipeline.inputs == []
    assert prepare_calls[0][2]["trim_long"] is False
    assert len(encoded_calls) == 1
    assert model.duration_estimator.calls == [("hello", "caller supplied transcript.", 3)]


def test_pipeline_text_only_skips_reference_audio_processing(monkeypatch):
    model, prepare_calls, encoded_calls = _build_fake_pipeline(
        monkeypatch,
        np.ones((1, _HOP_LENGTH), dtype=np.float32),
    )

    result = model.forward(_request("hello"))

    assert result.error is None
    assert prepare_calls == []
    assert encoded_calls == []
    assert model._asr_pipeline.inputs == []
    assert model.duration_estimator.calls == [("hello", "Nice to meet you.", 25)]
