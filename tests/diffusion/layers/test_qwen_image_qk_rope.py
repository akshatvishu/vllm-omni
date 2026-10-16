# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.layers import qwen_image_qk_rope as layer
from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer as qwen_model

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("rocm", "available", "triton", "architecture", "expected"),
    [
        (True, True, True, "gfx942:sramecc+:xnack-", True),
        (False, True, True, "gfx942", False),
        (True, False, True, "gfx942", False),
        (True, True, False, "gfx942", False),
        (True, True, True, "gfx950", False),
    ],
)
def test_hardware_gate(monkeypatch, rocm, available, triton, architecture, expected):
    monkeypatch.setattr(layer, "HAS_TRITON", triton)
    monkeypatch.setattr(layer, "current_omni_platform", SimpleNamespace(is_rocm=lambda: rocm))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(gcnArchName=architecture))
    assert layer.qwen_image_qk_rope_supported() is expected


def _inputs(text_length=17, device="cuda:0", random=False):
    factory = torch.randn if random else torch.empty
    image = factory((1, 4096, 9216), dtype=torch.bfloat16, device=device)
    text = factory((1, text_length, 9216), dtype=torch.bfloat16, device=device)
    weights = tuple(factory((128,), dtype=torch.bfloat16, device=device) for _ in range(4))
    # Extra coefficient rows exercise the supported prefix slicing contract.
    coeffs = tuple(factory((length + 3, 64), dtype=torch.complex64, device=device) for length in (4096, text_length))
    return image, text, *weights, *coeffs


def _attention(inputs):
    norms = [SimpleNamespace(weight=weight, eps=1e-6) for weight in inputs[2:6]]
    return SimpleNamespace(
        use_rocm_qk_rope=True,
        qk_norm=True,
        eps=1e-6,
        head_dim=128,
        query_num_heads=24,
        kv_num_heads=24,
        add_query_num_heads=24,
        add_kv_num_heads=24,
        parallel_config=None,
        norm_q=norms[0],
        norm_k=norms[1],
        norm_added_q=norms[2],
        norm_added_k=norms[3],
    )


def _supported(attn, inputs):
    return layer.can_use_qwen_image_qk_rope(attn, inputs[0], inputs[1], inputs[6], inputs[7])


@pytest.mark.cpu
@pytest.mark.parametrize("text_length", [1, 17, 128])
def test_supported_metadata(text_length):
    with FakeTensorMode():
        inputs = _inputs(text_length)
        attn = _attention(inputs)
        assert _supported(attn, inputs)
        attn.parallel_config = SimpleNamespace(sequence_parallel_size=1, tensor_parallel_size=1)
        assert _supported(attn, inputs)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "batch",
        "image_length",
        "text_empty",
        "text_long",
        "packed_width",
        "heads",
        "dtype",
        "cpu",
        "mixed_device",
        "strided_input",
        "no_norm",
        "eps",
        "norm_eps",
        "norm_missing",
        "weight_shape",
        "weight_dtype",
        "weight_device",
        "weight_stride",
        "coeff_shape",
        "coeff_short",
        "coeff_dtype",
        "coeff_device",
        "coeff_stride",
        "sp",
        "tp",
    ],
)
def test_unsupported_metadata_falls_back(case):
    with FakeTensorMode():
        inputs = list(_inputs())
        attn = _attention(inputs)
        if case == "disabled":
            attn.use_rocm_qk_rope = False
        elif case == "batch":
            inputs[0] = inputs[0].expand(2, -1, -1).contiguous()
        elif case == "image_length":
            inputs[0] = inputs[0][:, :4095]
        elif case in ("text_empty", "text_long"):
            inputs[1] = torch.empty(
                (1, 0 if case == "text_empty" else 129, 9216), dtype=torch.bfloat16, device="cuda:0"
            )
        elif case == "packed_width":
            inputs[0] = torch.empty((1, 4096, 6144), dtype=torch.bfloat16, device="cuda:0")
        elif case == "heads":
            attn.kv_num_heads = 12
        elif case == "dtype":
            inputs[0] = inputs[0].float()
        elif case == "cpu":
            inputs = list(_inputs(device="cpu"))
            attn = _attention(inputs)
        elif case == "mixed_device":
            inputs[1] = inputs[1].to("cuda:1")
        elif case == "strided_input":
            inputs[0] = torch.empty((1, 4096, 18432), dtype=torch.bfloat16, device="cuda:0")[..., ::2]
        elif case == "no_norm":
            attn.qk_norm = False
        elif case == "eps":
            attn.eps = 1e-5
        elif case == "norm_eps":
            attn.norm_added_k.eps = 1e-5
        elif case == "norm_missing":
            attn.norm_added_k = None
        elif case == "weight_shape":
            attn.norm_q.weight = inputs[2][:64]
        elif case == "weight_dtype":
            attn.norm_q.weight = inputs[2].float()
        elif case == "weight_device":
            attn.norm_q.weight = inputs[2].cpu()
        elif case == "weight_stride":
            attn.norm_q.weight = torch.empty(256, dtype=torch.bfloat16, device="cuda:0")[::2]
        elif case == "coeff_shape":
            inputs[6] = inputs[6][:, :32].contiguous()
        elif case == "coeff_short":
            inputs[6] = inputs[6][:4095]
        elif case == "coeff_dtype":
            inputs[6] = inputs[6].to(torch.complex128)
        elif case == "coeff_device":
            inputs[7] = inputs[7].cpu()
        elif case == "coeff_stride":
            inputs[6] = torch.empty((4099, 128), dtype=torch.complex64, device="cuda:0")[:, ::2]
        elif case in ("sp", "tp"):
            attn.parallel_config = SimpleNamespace(
                sequence_parallel_size=2 if case == "sp" else 1,
                tensor_parallel_size=2 if case == "tp" else 1,
            )
        assert not _supported(attn, inputs)


@pytest.mark.cpu
def test_fake_outputs_are_independent():
    with FakeTensorMode():
        inputs = _inputs()
        outputs = layer.qwen_image_qk_rope(*inputs)
        assert len(outputs) == 3
        for output in outputs:
            assert output.shape == (1, 4113, 24, 128)
            assert output.dtype == torch.bfloat16
            assert output.device == inputs[0].device
            assert output.is_contiguous()
        tensors = (*inputs, *outputs)
        for index, output in enumerate(outputs, start=len(inputs)):
            for other in tensors[:index]:
                assert not torch._C._is_alias_of(output, other)


@pytest.mark.cpu
def test_attention_forward_rejected_inputs_keep_original_buffers(monkeypatch):
    image = torch.arange(3 * 9216, dtype=torch.float32).reshape(1, 3, 9216)
    text = -torch.arange(2 * 9216, dtype=torch.float32).reshape(1, 2, 9216)
    inputs = (image, text, *(torch.ones(128) for _ in range(4)), torch.empty(3, 64), torch.empty(2, 64))
    attn = _attention(inputs)
    attn.to_qkv = lambda value: (image, None)
    attn.add_kv_proj = lambda value: (text, None)
    attn.to_out = attn.to_add_out = lambda value: value
    attn.rope = object()
    norm_calls = []

    def norm_rope(query, key, norm_q, norm_k, coeff, rope, eps, **kwargs):
        norm_calls.append((norm_q, norm_k, coeff))
        return query + 1, key + 2

    def forbidden_op(*args):
        pytest.fail("Rejected inputs must not launch the packed custom op")

    def attention(query, key, value, metadata):
        assert metadata is None
        for index, actual in enumerate((query, key, value)):
            expected = torch.cat(
                [packed.chunk(3, dim=-1)[index].reshape(1, -1, 24, 128) for packed in (text, image)], dim=1
            )
            if index < 2:
                expected = expected + index + 1
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        return value

    attn.attn = attention
    monkeypatch.setattr(qwen_model, "qwen_image_qk_rope", forbidden_op)
    monkeypatch.setattr(qwen_model, "_qwen_image_qk_norm_rope", norm_rope)
    image_out, text_out = qwen_model.QwenImageCrossAttention.forward(attn, image, text, inputs[6], inputs[7])
    assert len(norm_calls) == 2
    assert norm_calls[0][0] is attn.norm_q and norm_calls[0][2] is inputs[6]
    assert norm_calls[1][0] is attn.norm_added_q and norm_calls[1][2] is inputs[7]
    torch.testing.assert_close(image_out, image.chunk(3, dim=-1)[2], atol=0, rtol=0)
    torch.testing.assert_close(text_out, text.chunk(3, dim=-1)[2], atol=0, rtol=0)


@pytest.mark.cpu
def test_attention_forward_supported_inputs_use_custom_op(monkeypatch):
    with FakeTensorMode():
        inputs = _inputs()
        attn = _attention(inputs)
        attn.to_qkv = lambda value: (inputs[0], None)
        attn.add_kv_proj = lambda value: (inputs[1], None)
        attn.to_out = attn.to_add_out = lambda value: value
        outputs = tuple(inputs[0].new_empty((1, 4113, 24, 128)) for _ in range(3))
        calls = []

        def custom_op(*args):
            assert all(actual is expected for actual, expected in zip(args, inputs))
            calls.append("custom_op")
            return outputs

        def attention(query, key, value, metadata):
            assert all(actual is expected for actual, expected in zip((query, key, value), outputs))
            calls.append("attention")
            return value

        def forbidden_fallback(*args, **kwargs):
            pytest.fail("Supported inputs must bypass the separate normalization path")

        attn.attn = attention
        monkeypatch.setattr(qwen_model, "qwen_image_qk_rope", custom_op)
        monkeypatch.setattr(qwen_model, "_qwen_image_qk_norm_rope", forbidden_fallback)
        image_out, text_out = qwen_model.QwenImageCrossAttention.forward(
            attn, inputs[0], inputs[1], inputs[6], inputs[7]
        )
        assert calls == ["custom_op", "attention"]
        assert image_out.shape == (1, 4096, 3072)
        assert text_out.shape == (1, 17, 3072)


def _reference(inputs):
    streams = []
    for qkv, qw, kw, coeff in (
        (inputs[1], inputs[4], inputs[5], inputs[7]),
        (inputs[0], inputs[2], inputs[3], inputs[6]),
    ):
        q, k, v = (part.reshape(1, -1, 24, 128) for part in qkv.chunk(3, dim=-1))
        cos = coeff[: q.shape[1]].real.to(torch.bfloat16).float()[None, :, None, :]
        sin = coeff[: q.shape[1]].imag.to(torch.bfloat16).float()[None, :, None, :]
        rotated = []
        for value, weight in ((q, qw), (k, kw)):
            value = value.float()
            value = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)) * weight.float()
            value = value.to(torch.bfloat16).float()
            even, odd = value[..., ::2], value[..., 1::2]
            rotated.append(
                torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2).to(torch.bfloat16)
            )
        streams.append((*rotated, v))
    return tuple(torch.cat((streams[0][index], streams[1][index]), dim=1) for index in range(3))


def _require_gfx942():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("ROCm gfx942 required")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("ROCm gfx942 required")


@pytest.mark.cuda
@pytest.mark.parametrize("text_length", [1, 17, 128])
def test_numerical_reference_and_compile(text_length):
    _require_gfx942()
    torch.manual_seed(731)
    inputs = _inputs(text_length, random=True)
    eager = layer.qwen_image_qk_rope(*inputs)
    compiled = torch.compile(layer.qwen_image_qk_rope, fullgraph=True)(*inputs)
    expected = _reference(inputs)
    for index, (actual, compiled_actual, reference) in enumerate(zip(eager, compiled, expected)):
        # BF16 rounding and FMA/reduction ordering need not match eager bitwise.
        torch.testing.assert_close(actual, reference, atol=0 if index == 2 else 0.125, rtol=0 if index == 2 else 0.02)
        torch.testing.assert_close(compiled_actual, actual, atol=0, rtol=0)


@pytest.mark.cuda
def test_custom_op_contract():
    _require_gfx942()
    inputs = _inputs(random=True)
    torch.library.opcheck(torch.ops.vllm_omni.qwen_image_qk_rope.default, inputs)
