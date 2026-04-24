# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only unit tests for StableAudio VAE decode chunking logic.

These tests cover two layers:
  1. OmniDiffusionSamplingParams.vae_chunk_size field semantics.
  2. The chunk-size resolution logic that StableAudioPipeline.forward() uses,
     verified via a pure-Python mirror of that logic (no pipeline import needed).
"""

from __future__ import annotations

import pytest

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Pure-Python mirror of the pipeline's chunking decision
# ---------------------------------------------------------------------------


def _resolve_vae_calls(vae_chunk_size: int | None, batch_size: int) -> list[int]:
    """Return list of per-call batch sizes matching StableAudioPipeline.forward() logic."""
    if vae_chunk_size is not None and vae_chunk_size < 1:
        raise ValueError(f"`vae_chunk_size` must be >= 1, got {vae_chunk_size}")
    if vae_chunk_size is None or vae_chunk_size >= batch_size:
        return [batch_size]
    return [min(vae_chunk_size, batch_size - i) for i in range(0, batch_size, vae_chunk_size)]


# ---------------------------------------------------------------------------
# OmniDiffusionSamplingParams field tests
# ---------------------------------------------------------------------------


def test_vae_chunk_size_defaults_none():
    params = OmniDiffusionSamplingParams()
    assert params.vae_chunk_size is None


def test_vae_chunk_size_stored():
    params = OmniDiffusionSamplingParams(vae_chunk_size=2)
    assert params.vae_chunk_size == 2


def test_vae_chunk_size_clone_preserves_value():
    params = OmniDiffusionSamplingParams(vae_chunk_size=3)
    cloned = params.clone()
    assert cloned.vae_chunk_size == 3


def test_vae_chunk_size_clone_none_preserved():
    params = OmniDiffusionSamplingParams(vae_chunk_size=None)
    assert params.clone().vae_chunk_size is None


# ---------------------------------------------------------------------------
# Chunking logic tests
# ---------------------------------------------------------------------------


def test_none_produces_single_call():
    assert _resolve_vae_calls(None, batch_size=4) == [4]


def test_chunk_size_1_serializes_each_item():
    assert _resolve_vae_calls(1, batch_size=4) == [1, 1, 1, 1]


def test_chunk_size_2_splits_evenly():
    assert _resolve_vae_calls(2, batch_size=4) == [2, 2]


def test_chunk_size_2_uneven_batch():
    assert _resolve_vae_calls(2, batch_size=5) == [2, 2, 1]


def test_oversized_chunk_produces_single_call():
    assert _resolve_vae_calls(100, batch_size=4) == [4]


def test_chunk_size_equal_to_batch_produces_single_call():
    assert _resolve_vae_calls(4, batch_size=4) == [4]


def test_chunk_size_zero_raises():
    with pytest.raises(ValueError, match="vae_chunk_size"):
        _resolve_vae_calls(0, batch_size=4)


def test_chunk_size_negative_raises():
    with pytest.raises(ValueError, match="vae_chunk_size"):
        _resolve_vae_calls(-1, batch_size=4)


@pytest.mark.parametrize(
    "chunk_size,batch,expected_calls",
    [
        (None, 1, [1]),
        (1, 1, [1]),
        (1, 3, [1, 1, 1]),
        (2, 6, [2, 2, 2]),
        (3, 7, [3, 3, 1]),
        (10, 4, [4]),
    ],
)
def test_parametrized_chunking(chunk_size, batch, expected_calls):
    assert _resolve_vae_calls(chunk_size, batch) == expected_calls
