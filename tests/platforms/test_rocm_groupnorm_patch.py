import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch.nn as nn
import vllm.envs as vllm_envs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PATCH_PATH = Path(__file__).parents[2] / "vllm_omni" / "platforms" / "rocm" / "patch" / "worker" / "patch_groupnorm.py"
_SPEC = importlib.util.spec_from_file_location("test_patch_groupnorm", _PATCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
patch_groupnorm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = patch_groupnorm
_SPEC.loader.exec_module(patch_groupnorm)


def test_groupnorm_patch_respects_aiter_disabled(monkeypatch):
    group_norm = nn.GroupNorm(4, 8)
    model = nn.Module()
    model.vae = nn.Sequential(group_norm)

    monkeypatch.setattr(patch_groupnorm, "_original_initialize_model", lambda _: model)
    monkeypatch.setattr(vllm_envs, "VLLM_ROCM_USE_AITER", False)
    replace_called = False

    def replace_groupnorm(_):
        nonlocal replace_called
        replace_called = True
        return True

    monkeypatch.setattr(patch_groupnorm, "_replace_groupnorm_with_aiter", replace_groupnorm)

    assert patch_groupnorm._patched_initialize_model(None) is model
    assert not replace_called
    assert model.vae[0] is group_norm


def test_groupnorm_patch_keeps_aiter_path_enabled(monkeypatch):
    model = nn.Module()
    model.vae = nn.Sequential(nn.GroupNorm(4, 8))
    replace_called = False

    monkeypatch.setattr(patch_groupnorm, "_original_initialize_model", lambda _: model)
    monkeypatch.setattr(vllm_envs, "VLLM_ROCM_USE_AITER", True)
    fake_aiter_ops = types.ModuleType("vllm._aiter_ops")
    fake_aiter_ops.is_aiter_found_and_supported = lambda: True
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", fake_aiter_ops)

    def replace_groupnorm(_):
        nonlocal replace_called
        replace_called = True
        return True

    monkeypatch.setattr(patch_groupnorm, "_replace_groupnorm_with_aiter", replace_groupnorm)

    assert patch_groupnorm._patched_initialize_model(None) is model
    assert replace_called
