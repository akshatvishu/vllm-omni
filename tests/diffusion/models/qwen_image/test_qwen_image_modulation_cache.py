# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.compile as compile_module
import vllm_omni.diffusion.distributed.cfg_parallel as base_cfg_parallel_module
import vllm_omni.diffusion.models.qwen_image.cfg_parallel as cfg_parallel_module
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.hooks import HookRegistry
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import (
    _prepare_qwen_cfg_inputs,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QwenImagePipeline,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformerBlock,
    _apply_qwen_modulation_cache_hook,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _CountingModulation(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.calls = 0
        self.last_input_shape: tuple[int, ...] | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_input_shape = tuple(value.shape)
        return value + self.offset


class _CacheTestBlock(nn.Module):
    _compute_modulation_params = QwenImageTransformerBlock._compute_modulation_params

    def __init__(self, zero_cond_t: bool = False) -> None:
        super().__init__()
        self.zero_cond_t = zero_cond_t
        self.img_mod = _CountingModulation(1.0)
        self.txt_mod = _CountingModulation(2.0)

    def forward(
        self,
        *,
        temb: torch.Tensor,
        modulation_cache_key: torch.Tensor | None = None,
        modulation_params: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del modulation_cache_key
        if modulation_params is None:
            modulation_params = self._compute_modulation_params(temb)
        return modulation_params


class _CacheTestModel(nn.Module):
    _repeated_blocks = ["_CacheTestBlock"]
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_CacheTestBlock()])
        _apply_qwen_modulation_cache_hook(self.blocks[0])

    def forward(
        self,
        temb: torch.Tensor,
        modulation_cache_key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.blocks[0](
            temb=temb,
            modulation_cache_key=modulation_cache_key,
        )


def _call_block(
    block: _CacheTestBlock,
    temb: torch.Tensor,
    cache_key: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return block(
        temb=temb,
        modulation_cache_key=cache_key,
    )


def test_serial_cfg_pair_reuses_modulation_once() -> None:
    block = _CacheTestBlock()
    _apply_qwen_modulation_cache_hook(block)
    cache_key = torch.tensor(1)
    temb = torch.randn(1, 4)

    with torch.inference_mode():
        first = _call_block(block, temb, cache_key)
        second = _call_block(block, temb, cache_key)

    assert block.img_mod.calls == 1
    assert block.txt_mod.calls == 1
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])

    registry = HookRegistry.get_or_create(block)
    hook = registry.get_hook("qwen_cfg_modulation_cache")
    assert hook is not None
    assert hook._cache is None  # type: ignore[attr-defined]

    with torch.inference_mode():
        _call_block(block, temb, cache_key)

    assert block.img_mod.calls == 2
    assert block.txt_mod.calls == 2


def test_cache_miss_replaces_unmatched_entry() -> None:
    block = _CacheTestBlock()
    _apply_qwen_modulation_cache_hook(block)
    first_key = torch.tensor(1)
    second_key = torch.tensor(2)
    temb = torch.randn(1, 4)

    with torch.inference_mode():
        _call_block(block, temb, first_key)
        _call_block(block, temb, second_key)
        _call_block(block, temb, first_key)

    assert block.img_mod.calls == 3
    assert block.txt_mod.calls == 3


def test_missing_key_clears_unmatched_entry() -> None:
    block = _CacheTestBlock()
    _apply_qwen_modulation_cache_hook(block)
    cache_key = torch.tensor(1)
    temb = torch.randn(1, 4)

    with torch.inference_mode():
        _call_block(block, temb, cache_key)
        _call_block(block, temb, None)
        _call_block(block, temb, cache_key)

    assert block.img_mod.calls == 3
    assert block.txt_mod.calls == 3


def test_grad_enabled_calls_do_not_cache() -> None:
    block = _CacheTestBlock()
    _apply_qwen_modulation_cache_hook(block)
    cache_key = torch.tensor(1)
    temb = torch.randn(1, 4, requires_grad=True)

    _call_block(block, temb, cache_key)
    _call_block(block, temb, cache_key)

    assert block.img_mod.calls == 2
    assert block.txt_mod.calls == 2


def test_stream_capture_disables_and_clears_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    block = _CacheTestBlock()
    _apply_qwen_modulation_cache_hook(block)
    cache_key = torch.tensor(1)
    temb = torch.randn(1, 4)

    with torch.inference_mode():
        _call_block(block, temb, cache_key)

    registry = HookRegistry.get_or_create(block)
    hook = registry.get_hook("qwen_cfg_modulation_cache")
    assert hook is not None
    assert hook._cache is not None  # type: ignore[attr-defined]

    monkeypatch.setattr(hook, "_is_stream_capturing", lambda _temb: True)
    with torch.inference_mode():
        _call_block(block, temb, cache_key)

    assert block.img_mod.calls == 2
    assert block.txt_mod.calls == 2
    assert hook._cache is None  # type: ignore[attr-defined]


def test_modulation_cache_reset_between_requests() -> None:
    block = _CacheTestBlock()
    _apply_qwen_modulation_cache_hook(block)
    cache_key = torch.tensor(1)
    temb = torch.randn(1, 4)

    with torch.inference_mode():
        _call_block(block, temb, cache_key)

    registry = HookRegistry.get_or_create(block)
    registry.reset_hook("qwen_cfg_modulation_cache")

    with torch.inference_mode():
        _call_block(block, temb, cache_key)

    assert block.img_mod.calls == 2
    assert block.txt_mod.calls == 2


def test_zero_condition_text_modulation_uses_first_half() -> None:
    block = _CacheTestBlock(zero_cond_t=True)
    _apply_qwen_modulation_cache_hook(block)
    temb = torch.randn(4, 4)

    with torch.inference_mode():
        _call_block(block, temb, torch.tensor(1))

    assert block.img_mod.last_input_shape == (4, 4)
    assert block.txt_mod.last_input_shape == (2, 4)


def test_prepare_qwen_cfg_inputs_without_initialized_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        base_cfg_parallel_module,
        "is_cfg_group_initialized",
        lambda: False,
    )
    timestep = torch.tensor([500.0])

    model_timestep, cache_key = _prepare_qwen_cfg_inputs(timestep, True)

    torch.testing.assert_close(model_timestep, torch.tensor([0.5]))
    assert cache_key is model_timestep


@pytest.mark.parametrize(
    ("do_true_cfg", "cfg_world_size", "expect_cache_key"),
    [
        (True, 1, True),
        (True, 2, False),
        (False, 1, False),
    ],
)
def test_prepare_qwen_cfg_inputs(
    monkeypatch: pytest.MonkeyPatch,
    do_true_cfg: bool,
    cfg_world_size: int,
    expect_cache_key: bool,
) -> None:
    monkeypatch.setattr(
        cfg_parallel_module,
        "_get_cfg_world_size_or_one",
        lambda: cfg_world_size,
    )
    timestep = torch.tensor([500.0])

    model_timestep, cache_key = _prepare_qwen_cfg_inputs(timestep, do_true_cfg)

    torch.testing.assert_close(model_timestep, torch.tensor([0.5]))
    assert (cache_key is not None) is expect_cache_key
    if cache_key is not None:
        assert cache_key is model_timestep


def test_denoise_kwargs_share_pair_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cfg_parallel_module,
        "_get_cfg_world_size_or_one",
        lambda: 1,
    )
    latents = torch.randn(1, 2, 4)
    timestep = torch.tensor(500.0)
    prompt_embeds = torch.randn(1, 3, 4)
    prompt_mask = torch.ones(1, 3, dtype=torch.bool)

    positive, negative, output_slice = QwenImagePipeline._build_denoise_kwargs(
        None,
        latents=latents,
        timestep=timestep,
        guidance=None,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_mask,
        img_shapes=[(1, 1, 2)],
        txt_seq_lens=[3],
        do_true_cfg=True,
        negative_prompt_embeds=prompt_embeds.clone(),
        negative_prompt_embeds_mask=prompt_mask.clone(),
        negative_txt_seq_lens=[3],
    )

    assert negative is not None
    assert positive["timestep"] is negative["timestep"]
    assert positive["modulation_cache_key"] is negative["modulation_cache_key"]
    assert positive["modulation_cache_key"] is positive["timestep"]
    assert positive["modulation_cache_key"] is not None
    assert output_slice is None

    next_positive, next_negative, _ = QwenImagePipeline._build_denoise_kwargs(
        None,
        latents=latents,
        timestep=timestep,
        guidance=None,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_mask,
        img_shapes=[(1, 1, 2)],
        txt_seq_lens=[3],
        do_true_cfg=True,
        negative_prompt_embeds=prompt_embeds.clone(),
        negative_prompt_embeds_mask=prompt_mask.clone(),
        negative_txt_seq_lens=[3],
    )

    assert next_negative is not None
    assert next_positive["modulation_cache_key"] is next_negative["modulation_cache_key"]
    assert next_positive["modulation_cache_key"] is not positive["modulation_cache_key"]


def test_modulation_cache_hook_stays_outside_regional_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _CacheTestModel()
    compiled_calls = 0

    def fake_compile(function: Any, *args: Any, **kwargs: Any):
        del args, kwargs

        def compiled(*args: Any, **kwargs: Any):
            nonlocal compiled_calls
            compiled_calls += 1
            return function(*args, **kwargs)

        return compiled

    monkeypatch.setattr(compile_module.torch, "compile", fake_compile)
    regionally_compile(model)

    cache_key = torch.tensor(1)
    temb = torch.randn(1, 4)
    with torch.inference_mode():
        first = model(temb, cache_key)
        second = model(temb, cache_key)

    assert compiled_calls == 2
    assert model.blocks[0].img_mod.calls == 1
    assert model.blocks[0].txt_mod.calls == 1
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
