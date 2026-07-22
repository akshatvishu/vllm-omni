# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiniCPM-o 4.5 Talker and Token2Wav streaming."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _build_stream_chunks,
    _generate_reference_tts_tokens,
    _iter_tts_condition_chunks,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_tts_condition_chunks_use_reference_schedule() -> None:
    embeds = torch.arange(25 * 2, dtype=torch.float32).reshape(25, 2)

    chunks = list(_iter_tts_condition_chunks(embeds))

    assert [chunk.shape for chunk, _ in chunks] == [(1, 10, 2), (1, 10, 2), (1, 5, 2)]
    assert [is_finished for _, is_finished in chunks] == [False, False, True]
    assert torch.equal(torch.cat([chunk.squeeze(0) for chunk, _ in chunks]), embeds)


def test_tts_condition_chunks_mark_empty_text_finished() -> None:
    chunks = list(_iter_tts_condition_chunks(torch.empty(0, 2)))

    assert len(chunks) == 1
    assert chunks[0][0].shape == (1, 0, 2)
    assert chunks[0][1] is True


def test_reference_tts_tokens_preserve_generator_state_and_flush_buffer() -> None:
    calls = []

    class FakeSamplingParams:
        temperature = 0.8
        top_p = 0.85
        top_k = 25
        repetition_penalty = 1.05

    class FakeGenerator:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            self._token_buffer = []
            self.call_index = 0

        def generate_with_buffer(self, *, condition, text_finished):
            calls.append(("condition", condition.clone(), text_finished))
            self.call_index += 1
            if text_finished:
                self._token_buffer.append(torch.tensor([[99]], dtype=torch.long))
            yield torch.full((1, 2), self.call_index, dtype=torch.long), False

    def fake_gen_logits(**kwargs):
        calls.append(("logits", kwargs))
        return ["warper"], ["processor"]

    fake_module = SimpleNamespace(
        TTSSamplingParams=FakeSamplingParams,
        TTSStreamingGenerator=FakeGenerator,
        gen_logits=fake_gen_logits,
    )
    fake_tts = SimpleNamespace(
        config=SimpleNamespace(num_audio_tokens=100),
        device=torch.device("cpu"),
    )

    generated = _generate_reference_tts_tokens(
        fake_tts,
        torch.zeros(25, 4),
        tts_module=fake_module,
    )

    assert torch.equal(generated, torch.tensor([[1, 1, 2, 2, 3, 3, 99]]))
    condition_calls = [call for call in calls if call[0] == "condition"]
    assert [call[2] for call in condition_calls] == [False, False, True]
    assert [call[1].shape for call in condition_calls] == [(1, 10, 4), (1, 10, 4), (1, 5, 4)]
    init_call = next(call for call in calls if call[0] == "init")
    assert init_call[1]["chunk_size"] == 25
    assert init_call[1]["temperature"] == 0.8
    logits_call = next(call for call in calls if call[0] == "logits")
    assert logits_call[1] == {
        "num_code": 100,
        "repetition_penalty": 1.05,
        "top_p": 0.85,
        "top_k": 25,
    }


def test_stream_chunks_overlap_lookahead_and_flush_tail() -> None:
    chunks = _build_stream_chunks(list(range(120)), chunk_size=50)

    assert len(chunks) == 3
    assert chunks[0][0] == [4218, 4218, 4218] + list(range(50))
    assert chunks[1][0] == list(range(47, 100))
    assert chunks[2][0] == list(range(97, 120))
    assert chunks[0][1] is False
    assert chunks[1][1] is False
    assert chunks[2][1] is True


def test_stream_chunks_preserve_terminal_tokens_on_exact_boundary() -> None:
    chunks = _build_stream_chunks(list(range(100)), chunk_size=50)

    assert chunks[-1][1] is True
    assert chunks[-1][0] == list(range(97, 100))
    assert chunks[-2][0][-1] == 99


def test_stream_chunks_match_long_form_schedule() -> None:
    chunks = _build_stream_chunks(list(range(3206)), chunk_size=50)

    assert len(chunks) == 65
    assert all(len(token_chunk) == 53 for token_chunk, is_last in chunks[:-1] if not is_last)
    assert len(chunks[-1][0]) == 9
    assert chunks[-1][1] is True


@pytest.mark.parametrize(
    ("chunk_size", "lookahead"),
    [(0, 3), (-1, 3), (50, -1)],
)
def test_stream_chunks_reject_invalid_configuration(chunk_size: int, lookahead: int) -> None:
    with pytest.raises(ValueError):
        _build_stream_chunks([1, 2, 3], chunk_size=chunk_size, lookahead=lookahead)
