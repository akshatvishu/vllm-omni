# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers import BatchEncoding

from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    _prepare_qwen21_decode_varlen,
    _prepare_qwen21_decode_varlen_host,
)
from vllm_omni.diffusion.worker.input_batch import _prepare_prompt_embeds
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "rows,slots",
    [
        ([[1, 1, 0], [1, 0, 0]], [0, 0, 0, 1]),
        ([[0, 1, 1], [0, 0, 1]], [0, 0, 0, 1]),
        ([[1, 0, 1], [0, 1, 0]], [0, 0, 0, 1]),
        ([[0, 0, 0], [1, 1, 1]], [0, 0, 0, 1]),
        ([[1, 1, 1]], [0, 0, 0, 1]),
        ([[1, 1, 0, 1], [0, 0, 1, 0]], [0, 1, 0, 0, 1]),
    ],
    ids=["right-padding", "left-padding", "holes", "empty-text-row", "all-visible", "interleaved-image"],
)
def test_host_metadata_matches_explicit_joint_validity(rows, slots):
    encoder_mask = torch.tensor(rows, dtype=torch.bool)
    img_mask = torch.tensor([slots] * len(rows), dtype=torch.bool)
    # Scalar reference independently expands image slots and preserves each text bit.
    expected_rows = []
    for row in rows:
        expanded = []
        for index, image_slot in enumerate(slots):
            if image_slot:
                expanded.extend([True] * 4)
            else:
                expanded.append(bool(row[index]))
        expected_rows.append(expanded)
    expected = _prepare_qwen21_decode_varlen(torch.tensor(expected_rows), 4)
    actual = _prepare_qwen21_decode_varlen_host(encoder_mask, img_mask, 4, torch.device("cpu"))
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert encoder_mask.tolist() == [[bool(x) for x in row] for row in rows]
    assert img_mask.tolist() == [[bool(x) for x in slots]] * len(rows)


def test_host_metadata_rebuilds_after_same_shape_mutation_and_row_reorder():
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    slots = torch.tensor([[0, 0, 0, 1]] * 2, dtype=torch.bool)
    first = _prepare_qwen21_decode_varlen_host(mask, slots, 4, torch.device("cpu"))
    old_indices = first.indices_k.clone()
    mask.copy_(torch.tensor([[0, 1, 0], [1, 1, 0]], dtype=torch.bool))
    second = _prepare_qwen21_decode_varlen_host(mask, slots, 4, torch.device("cpu"))
    torch.testing.assert_close(first.indices_k, old_indices, rtol=0, atol=0)
    assert first.cu_seqlens_k.tolist() == [0, 6, 11]
    assert second.cu_seqlens_k.tolist() == [0, 5, 11]
    assert not torch.equal(first.indices_k, second.indices_k)
    reversed_rows = _prepare_qwen21_decode_varlen_host(mask.flip(0), slots, 4, torch.device("cpu"))
    assert reversed_rows.cu_seqlens_k.tolist() == [0, 6, 11]
    # Removing one row retains its padded prefix width, not its valid-text count.
    shrunk = _prepare_qwen21_decode_varlen_host(mask[:1], slots[:1], 4, torch.device("cpu"))
    assert shrunk.indices_k.tolist() == [1, 3, 4, 5, 6]


@pytest.mark.parametrize(
    "exclusion", [None, "edit", "not_hip", "cpu", "fp32", "sdpa", "no_backend", "sp", "fp8", "grad", "compile"]
)
def test_pipeline_host_mask_policy(monkeypatch, exclusion):
    pipeline = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu" if exclusion == "cpu" else "cuda")
    backend = SDPABackend if exclusion == "sdpa" else None if exclusion == "no_backend" else FlashAttentionBackend
    pipeline.transformer = SimpleNamespace(
        transformer_blocks=[SimpleNamespace(attn=SimpleNamespace(attn=SimpleNamespace(attn_backend=backend)))],
        parallel_config=DiffusionParallelConfig(ulysses_degree=2) if exclusion == "sp" else None,
        prefix_kv_cache_dtype="fp8" if exclusion == "fp8" else None,
    )
    monkeypatch.setattr(torch.version, "hip", None if exclusion == "not_hip" else "test")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: exclusion == "compile")
    with torch.set_grad_enabled(exclusion == "grad"):
        actual = pipeline._use_host_prompt_masks(
            has_images=exclusion == "edit", dtype=torch.float32 if exclusion == "fp32" else torch.bfloat16
        )
    assert actual is (exclusion is None)


@torch.no_grad()
def test_host_prompt_producer_preserves_compaction_and_padding(monkeypatch):
    pipeline = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.prompt_template_t2i = "{}"
    pipeline._max_length = 16
    pipeline._drop_idx = 1
    pipeline._img_token_id = 99
    ids = torch.tensor([[0, 0, 10, 11, 12], [0, 10, 21, 22, 23]])
    validity = ids != 0
    processor = Mock(side_effect=lambda **kwargs: BatchEncoding(dict(input_ids=ids, attention_mask=validity)))
    processor.tokenizer = Mock(return_value=BatchEncoding(dict(attention_mask=validity)))
    pipeline.processor = processor
    hidden = torch.arange(40, dtype=torch.bfloat16).reshape(2, 5, 4)
    pipeline.text_encoder = Mock(
        spec=nn.Module,
        dtype=torch.bfloat16,
        model=SimpleNamespace(norm=nn.Identity()),
        return_value=SimpleNamespace(hidden_states=[hidden]),
    )
    monkeypatch.setattr(pipeline, "_use_host_prompt_masks", lambda *args: False)
    original = pipeline._get_qwen_prompt_embeds(["short", "long"])
    monkeypatch.setattr(pipeline, "_use_host_prompt_masks", lambda *args: True)
    candidate = pipeline._get_qwen_prompt_embeds(["short", "long"])
    for actual, expected in zip(candidate, original):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert candidate[1].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert candidate[2].tolist() == [[False] * 3] * 2
    torch.testing.assert_close(candidate[0][0, :2], hidden[0, 3:])
    torch.testing.assert_close(candidate[0][1], hidden[1, 2:])


def test_host_masks_follow_batch_order_and_baked_prefix_width():
    states = [
        StepRequestState(
            request_id=str(length),
            sampling=OmniDiffusionSamplingParams(),
            prompt_embeds=torch.full((1, length, 4), float(length)),
            prompt_embeds_mask=torch.ones(1, length, dtype=torch.bool),
            txt_seq_lens=[length],
        )
        for length in (2, 3)
    ]
    for state in states:
        state.extra["image_pad_mask"] = torch.zeros_like(state.prompt_embeds_mask)
        state.extra["prefill_prompt_seq_len"] = 3

    _, mask = _prepare_prompt_embeds(states)
    assert mask.device.type == "cpu"
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    _, reordered = _prepare_prompt_embeds(states[::-1])
    assert reordered.tolist() == mask.flip(0).tolist()

    _, shrunk = _prepare_prompt_embeds(states[:1])
    assert shrunk.shape == (1, 2)
    # The pipeline restores the width baked into the prefill cache after a
    # longer request leaves the batch. Mirror those existing production calls.
    width = QwenImage21Pipeline._resolve_prompt_seq_len(states[:1], shrunk.shape[1])
    assert width == 3
    shrunk = nn.functional.pad(shrunk, (0, width - shrunk.shape[1]))
    slots = QwenImage21Pipeline._append_target_slots(
        QwenImage21Pipeline._pad_rows_to_seq_len([states[0].extra["image_pad_mask"]], width), 4
    )
    prepared = _prepare_qwen21_decode_varlen_host(shrunk, slots, 4, torch.device("cpu"))
    assert prepared.indices_k.tolist() == [0, 1, 3, 4, 5, 6]
