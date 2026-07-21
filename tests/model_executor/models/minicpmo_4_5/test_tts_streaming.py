# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiniCPM-o 4.5 Token2Wav streaming chunk construction."""

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _build_stream_chunks

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
