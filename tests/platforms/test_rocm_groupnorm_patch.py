# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PATCH_PATH = Path(__file__).parents[2] / "vllm_omni" / "platforms" / "rocm" / "patch" / "worker" / "patch_groupnorm.py"
_SPEC = importlib.util.spec_from_file_location("test_patch_groupnorm", _PATCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
patch_groupnorm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = patch_groupnorm
_SPEC.loader.exec_module(patch_groupnorm)


def _install_fake_aiter(monkeypatch, calls, output=None):
    class FakeAiterGroupNorm(nn.GroupNorm):
        def forward(self, input):
            calls.append((input, self.weight, self.bias))
            return torch.empty_like(input) if output is None else output

    fake_groupnorm = types.ModuleType("aiter.ops.groupnorm")
    fake_groupnorm.GroupNorm = FakeAiterGroupNorm
    monkeypatch.setitem(sys.modules, "aiter", types.ModuleType("aiter"))
    monkeypatch.setitem(sys.modules, "aiter.ops", types.ModuleType("aiter.ops"))
    monkeypatch.setitem(sys.modules, "aiter.ops.groupnorm", fake_groupnorm)


def test_groupnorm_autocast_casts_kernel_input_to_fp32(monkeypatch):
    calls = []
    _install_fake_aiter(monkeypatch, calls)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda _: True)

    vae = nn.Sequential(nn.GroupNorm(32, 1024, eps=1e-6))
    assert patch_groupnorm._replace_groupnorm_with_aiter(vae)
    input = torch.randn(1, 1024, 1, 4, 4, dtype=torch.float16)

    output = vae(input)

    kernel_input, weight, bias = calls.pop()
    assert kernel_input.dtype == weight.dtype == bias.dtype == torch.float32
    assert output.dtype == torch.float32


def test_groupnorm_same_dtype_uses_aiter(monkeypatch):
    expected = torch.randn(1, 8, 2, 2)
    calls = []
    _install_fake_aiter(monkeypatch, calls, output=expected)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda _: False)

    vae = nn.Sequential(nn.GroupNorm(4, 8))
    assert patch_groupnorm._replace_groupnorm_with_aiter(vae)

    assert vae(torch.randn(1, 8, 2, 2)) is expected
    assert len(calls) == 1


def test_groupnorm_mixed_dtype_without_autocast_uses_torch(monkeypatch):
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda _: False)
    calls = []
    _install_fake_aiter(monkeypatch, calls)

    input = torch.randn(1, 8, 2, 2, dtype=torch.float16)
    vae = nn.Sequential(nn.GroupNorm(4, 8, dtype=torch.float32))
    weight = vae[0].weight
    bias = vae[0].bias
    assert weight is not None and bias is not None
    assert patch_groupnorm._replace_groupnorm_with_aiter(vae)

    actual = vae(input)
    expected = torch.nn.functional.group_norm(input, 4, weight, bias, 1e-5)

    torch.testing.assert_close(actual, expected)
    assert not calls


def test_replace_groupnorm_preserves_parameters(monkeypatch):
    _install_fake_aiter(monkeypatch, [])
    original = nn.GroupNorm(4, 8, eps=1e-6)
    vae = nn.Sequential(original)

    assert patch_groupnorm._replace_groupnorm_with_aiter(vae)
    assert isinstance(vae[0], patch_groupnorm._AiterGroupNormAutocastMixin)
    assert vae[0].weight is original.weight
    assert vae[0].bias is original.bias
    assert vae[0].eps == original.eps
