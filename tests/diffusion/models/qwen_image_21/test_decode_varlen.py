# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.attention.backends.utils import fa
from vllm_omni.diffusion.attention.selector import _load_backend_cls
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _CudaDeviceTensor(torch.Tensor):
    """Exercise the ROCm eligibility gate with CPU storage and mocked blocks."""

    @property
    def device(self):
        return torch.device("cuda")


@pytest.fixture
def decode_model(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(transformer, "is_forward_context_available", lambda: False)

    def make(num_blocks=3, backend=FlashAttentionBackend, requested="FLASH_ATTN", dtype=torch.bfloat16):
        model = transformer.QwenImage21Transformer2DModel.__new__(transformer.QwenImage21Transformer2DModel)
        nn.Module.__init__(model)
        model.causal_condition = True
        model.causal_block = True
        model.parallel_config = None
        model.prefix_kv_cache_dtype = None
        model._decode_graph_manager = None
        target = torch.zeros(2, 4, 8, dtype=dtype).as_subclass(_CudaDeviceTensor)
        model.sequence_prepare = Mock(return_value=(target[:, :0], target, torch.empty(0, 4), torch.empty(4, 4)))
        model.sequence_prepare.expand_image_pad_mask = transformer.QwenImage21SequencePrepare.expand_image_pad_mask
        model.time_text_embed = Mock(return_value=torch.zeros(3, 8, dtype=dtype))
        model.modulation = Mock(return_value=torch.zeros(3, 32, dtype=dtype))
        model.norm_out = Mock(side_effect=lambda values, *args: values)
        model.proj_out = nn.Identity()
        model.transformer_blocks = [
            Mock(
                attn=SimpleNamespace(attn=SimpleNamespace(attn_backend=backend, backend_pref=requested)),
                side_effect=lambda hidden_states, **kwargs: hidden_states.as_subclass(torch.Tensor),
            )
            for _ in range(num_blocks)
        ]
        inputs = dict(
            hidden_states=torch.zeros(2, 4, 8, dtype=dtype),
            encoder_hidden_states=None,
            timestep=torch.ones(2),
            img_shapes=[[(1, 2, 2)], [(1, 2, 2)]],
            img_mask=torch.tensor([[False, False, False, True]] * 2),
            encoder_hidden_states_mask=torch.tensor([[True, False, True, True], [False, True, False, True]]),
            kv_cache=[{"cond": {"key": torch.empty(2, 3, 1, 8)}} for _ in range(num_blocks)],
        )
        return model, inputs

    return make


@pytest.mark.parametrize("num_blocks", [1, 3, 32])
@torch.no_grad()
def test_decode_prepares_metadata_once_per_forward_and_refreshes_changed_mask(monkeypatch, decode_model, num_blocks):
    model, inputs = decode_model(num_blocks=num_blocks)
    prepare = Mock(wraps=fa._get_unpad_data)
    monkeypatch.setattr(fa, "_get_unpad_data", prepare)

    model(**inputs)
    first_metadata = model.transformer_blocks[0].call_args.kwargs["attn_metadata"]
    first = first_metadata.extra["qwen21_decode_varlen"]
    assert prepare.call_count == 1
    for block in model.transformer_blocks:
        assert block.call_args.kwargs["attn_metadata"] is first_metadata
    assert first_metadata.extra["attention_mask_mode"] == "padding"
    torch.testing.assert_close(first.indices_k, torch.tensor([0, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13]))
    torch.testing.assert_close(first.cu_seqlens_k, torch.tensor([0, 6, 11], dtype=torch.int32))
    torch.testing.assert_close(first.cu_seqlens_q, torch.tensor([0, 4, 8], dtype=torch.int32))
    assert (first.max_seqlen_k, first.max_seqlen_q) == (6, 4)

    # Reuse and mutate the same mask object, with both changed positions and lengths.
    inputs["encoder_hidden_states_mask"][:, :3] = torch.tensor([[False, True, False], [True, True, False]])
    model(**inputs)
    second_metadata = model.transformer_blocks[0].call_args.kwargs["attn_metadata"]
    second = second_metadata.extra["qwen21_decode_varlen"]
    assert prepare.call_count == 2
    assert second is not first
    for block in model.transformer_blocks:
        assert block.call_args.kwargs["attn_metadata"] is second_metadata
    torch.testing.assert_close(second.indices_k, torch.tensor([1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]))
    torch.testing.assert_close(second.cu_seqlens_k, torch.tensor([0, 5, 11], dtype=torch.int32))
    torch.testing.assert_close(first.cu_seqlens_k, torch.tensor([0, 6, 11], dtype=torch.int32))


@pytest.mark.parametrize("requested", ["TORCH_SDPA", "FLASH_ATTN"])
@torch.no_grad()
def test_resolved_sdpa_bypasses_decode_preparation(monkeypatch, decode_model, requested):
    # vLLM queries the architecture while importing its ROCm platform module.
    monkeypatch.setattr("torch.cuda.get_device_properties", lambda device: SimpleNamespace(gcnArchName="gfx942"))
    from vllm_omni.platforms.rocm.platform import RocmOmniPlatform

    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (9, 4))
    monkeypatch.setattr("vllm._aiter_ops.is_aiter_found_and_supported", lambda: False)
    backend_path = RocmOmniPlatform.get_diffusion_attn_backend_cls(requested, head_size=8)
    backend = _load_backend_cls(backend_path)
    assert backend is SDPABackend
    model, inputs = decode_model(backend=backend, requested=requested)
    prepare = Mock(side_effect=AssertionError("SDPA must not prepare FlashAttention metadata"))
    monkeypatch.setattr(fa, "_get_unpad_data", prepare)
    model(**inputs)
    prepare.assert_not_called()
    for block in model.transformer_blocks:
        metadata = block.call_args.kwargs["attn_metadata"]
        assert metadata.attn_mask is not None
        assert "qwen21_decode_varlen" not in metadata.extra
        assert "attention_mask_mode" not in metadata.extra

    expected_mask = torch.tensor(
        [[True, False, True, True, True, True, True], [False, True, False, True, True, True, True]]
    )
    query = torch.zeros(2, 4, 1, 1)
    key = torch.zeros(2, 7, 1, 1)
    value = torch.arange(14, dtype=torch.float32).reshape(2, 7, 1, 1)
    impl = SDPAImpl(num_heads=1, head_size=1, softmax_scale=1.0)
    actual = impl.forward_hip(query, key, value, metadata)
    expected = nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=expected_mask[:, None, None, :],
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "exclusion", ["not_hip", "cpu", "fp32", "prefix_fp8", "sp", "grad", "compile", "unpadded", "no_backend"]
)
def test_decode_preparation_exclusions(monkeypatch, decode_model, exclusion):
    model, inputs = decode_model(
        backend=None if exclusion == "no_backend" else FlashAttentionBackend,
        dtype=torch.float32 if exclusion == "fp32" else torch.bfloat16,
    )
    if exclusion == "not_hip":
        monkeypatch.setattr(torch.version, "hip", None)
    elif exclusion == "cpu":
        values = model.sequence_prepare.return_value
        model.sequence_prepare.return_value = tuple(value.as_subclass(torch.Tensor) for value in values)
    elif exclusion == "prefix_fp8":
        model.prefix_kv_cache_dtype = "fp8_e4m3"
    elif exclusion == "sp":
        model.parallel_config = DiffusionParallelConfig(ulysses_degree=2)
    elif exclusion == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif exclusion == "unpadded":
        inputs["encoder_hidden_states_mask"].fill_(True)
    prepare = Mock(side_effect=AssertionError("ineligible forward prepared decode metadata"))
    monkeypatch.setattr(fa, "_get_unpad_data", prepare)
    with torch.set_grad_enabled(exclusion == "grad"):
        model(**inputs)
    prepare.assert_not_called()


@torch.no_grad()
def test_prefill_does_not_prepare_decode_metadata(monkeypatch, decode_model):
    model, inputs = decode_model(num_blocks=32)
    _, target, _, freqs = model.sequence_prepare.return_value
    model.sequence_prepare.return_value = (target[:, :3], target, freqs[:3], freqs)
    model.txt_in = nn.Identity()
    inputs["encoder_hidden_states"] = torch.zeros(2, 3, 8, dtype=torch.bfloat16)
    inputs["kv_cache"] = [{} for _ in model.transformer_blocks]
    prepare = Mock(side_effect=AssertionError("prefill must not prepare decode metadata"))
    monkeypatch.setattr(fa, "_get_unpad_data", prepare)
    model(**inputs)
    prepare.assert_not_called()
    for block in model.transformer_blocks:
        metadata = block.call_args.kwargs["attn_metadata"]
        assert metadata.attn_mask.ndim == 4
        assert "qwen21_decode_varlen" not in metadata.extra
        assert metadata.extra["attention_mask_mode"] == "arbitrary"
        assert metadata is model.transformer_blocks[0].call_args.kwargs["attn_metadata"]


@pytest.mark.parametrize(
    "exclusion",
    [
        "not_hip",
        "cpu",
        "fp32",
        "prefix_fp8",
        "sp",
        "grad",
        "compile",
        "unpadded",
        "no_backend",
        "sdpa",
        "piecewise",
        "noncausal",
    ],
)
def test_prefill_mask_publication_exclusions(monkeypatch, decode_model, exclusion):
    backend = None if exclusion == "no_backend" else SDPABackend if exclusion == "sdpa" else FlashAttentionBackend
    model, inputs = decode_model(backend=backend, dtype=torch.float32 if exclusion == "fp32" else torch.bfloat16)
    _, target, _, freqs = model.sequence_prepare.return_value
    if exclusion == "cpu":
        target = target.as_subclass(torch.Tensor)
    model.sequence_prepare.return_value = (target[:, :3], target, freqs[:3], freqs)
    model.txt_in = nn.Identity()
    inputs["encoder_hidden_states"] = torch.zeros(2, 3, 8, dtype=target.dtype)
    inputs["kv_cache"] = [{} for _ in model.transformer_blocks]
    inputs["attention_kwargs"] = {"attn_path": "mask"}
    if exclusion == "not_hip":
        monkeypatch.setattr(torch.version, "hip", None)
    elif exclusion == "prefix_fp8":
        model.prefix_kv_cache_dtype = "fp8_e4m3"
    elif exclusion == "sp":
        model.parallel_config = DiffusionParallelConfig(ulysses_degree=2)
    elif exclusion == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif exclusion == "unpadded":
        inputs["encoder_hidden_states_mask"].fill_(True)
    elif exclusion == "piecewise":
        inputs["attention_kwargs"] = {"attn_path": "piecewise"}
    elif exclusion == "noncausal":
        model.causal_block = False
    with torch.set_grad_enabled(exclusion == "grad"):
        model(**inputs)
    for block in model.transformer_blocks:
        metadata = block.call_args.kwargs["attn_metadata"]
        assert "attention_mask_mode" not in metadata.extra


def _reference_varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    outputs = []
    for index in range(cu_seqlens_q.numel() - 1):
        query = q[cu_seqlens_q[index] : cu_seqlens_q[index + 1]].transpose(0, 1)
        key = k[cu_seqlens_k[index] : cu_seqlens_k[index + 1]].transpose(0, 1)
        value = v[cu_seqlens_k[index] : cu_seqlens_k[index + 1]].transpose(0, 1)
        outputs.append(nn.functional.scaled_dot_product_attention(query, key, value).transpose(0, 1))
    return torch.cat(outputs)


@pytest.mark.parametrize("case", ["eligible", "not_hip", "fp32", "query_length", "batch_size"])
@torch.no_grad()
def test_flash_decode_reuses_metadata_and_matches_existing_unpadding(monkeypatch, case):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    kernel = Mock(side_effect=_reference_varlen)
    monkeypatch.setattr(fa, "flash_attn_varlen_func", kernel)
    impl = FlashAttentionImpl(num_heads=2, head_size=8, softmax_scale=8**-0.5)
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(2, 4, 2, 8, generator=generator).bfloat16()
    key = torch.randn(2, 7, 2, 8, generator=generator).bfloat16()
    value = torch.randn(2, 7, 2, 8, generator=generator).bfloat16()
    if case == "fp32":
        query, key, value = query.float(), key.float(), value.float()
    elif case == "not_hip":
        monkeypatch.setattr(torch.version, "hip", None)
    mask = torch.tensor([[True, False, True, True, True, True, True], [False, True, False, True, True, True, True]])
    metadata = AttentionMetadata(attn_mask=mask)
    expected = impl.forward_cuda(query, key, value, metadata)
    reference_call = kernel.call_args
    prepared = transformer._prepare_qwen21_decode_varlen(mask, query.shape[1])
    if case == "query_length":
        prepared = prepared._replace(max_seqlen_q=query.shape[1] + 1)
    elif case == "batch_size":
        prepared = prepared._replace(cu_seqlens_q=prepared.cu_seqlens_q[:2])
    metadata.extra.update(qwen21_decode_varlen=prepared, attention_mask_mode="padding")
    unpad = Mock(wraps=fa._unpad_input)
    monkeypatch.setattr(fa, "_unpad_input", unpad)
    for _ in range(3):
        actual = impl.forward_cuda(query, key, value, metadata)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_tensor, expected_tensor in zip(kernel.call_args.args, reference_call.args):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        for name in ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"):
            torch.testing.assert_close(kernel.call_args.kwargs[name], reference_call.kwargs[name], rtol=0, atol=0)
        if case == "eligible":
            assert kernel.call_args.kwargs["cu_seqlens_q"] is prepared.cu_seqlens_q
            assert kernel.call_args.kwargs["cu_seqlens_k"] is prepared.cu_seqlens_k
    assert unpad.call_count == (0 if case == "eligible" else 3)


@torch.no_grad()
def test_mismatched_mask_shape_bypasses_prepared_metadata(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    kernel = Mock(side_effect=AssertionError("invalid mask must not reach attention"))
    monkeypatch.setattr(fa, "flash_attn_varlen_func", kernel)
    impl = FlashAttentionImpl(num_heads=2, head_size=8, softmax_scale=8**-0.5)
    query = torch.zeros(2, 4, 2, 8, dtype=torch.bfloat16)
    key = torch.zeros(2, 7, 2, 8, dtype=torch.bfloat16)
    valid_mask = torch.ones(2, 7, dtype=torch.bool)
    prepared = transformer._prepare_qwen21_decode_varlen(valid_mask, query.shape[1])
    invalid_mask = torch.ones(2, 8, dtype=torch.bool)
    unpad = Mock(wraps=fa._unpad_input)
    monkeypatch.setattr(fa, "_unpad_input", unpad)

    # The original path cannot unpad this invalid mask either. Verify the guard
    # preserves that path instead of silently accepting unrelated prepared data.
    for metadata in (None, prepared):
        unpad.reset_mock()
        with pytest.raises(IndexError):
            impl._forward_varlen_masked(query, key, key, invalid_mask, metadata)
        unpad.assert_called_once_with(key, invalid_mask)
    kernel.assert_not_called()
