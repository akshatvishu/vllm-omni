# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from vllm_omni.diffusion.model_loader.gguf_adapters import get_gguf_adapter
from vllm_omni.diffusion.model_loader.gguf_adapters.wan22 import Wan22GGUFAdapter


def _make_od_config(model_class_name: str = "WanPipeline", model_type: str = "wan"):
    return SimpleNamespace(
        model_class_name=model_class_name,
        tf_model_config={"model_type": model_type},
    )


def _make_source(prefix: str = "", subfolder: str = "transformer"):
    return SimpleNamespace(prefix=prefix, subfolder=subfolder)


class _FakeTransformer:
    pass


def test_wan_adapter_selected_for_wan_models():
    adapter = get_gguf_adapter(
        "dummy.gguf",
        _FakeTransformer(),
        _make_source(),
        _make_od_config(),
    )
    assert isinstance(adapter, Wan22GGUFAdapter)


def test_wan_adapter_maps_qkv_ffn_and_output_names():
    mapper = Wan22GGUFAdapter.gguf_to_hf_mapper

    assert mapper.apply_list(["blocks.0.attn1.to_q.weight"]) == [
        "blocks.0.attn1.to_qkv.weight"
    ]
    assert mapper.apply_list(["blocks.0.attn1.to_out.0.weight"]) == [
        "blocks.0.attn1.to_out.weight"
    ]
    assert mapper.apply_list(["blocks.0.ffn.net.0.proj.weight"]) == [
        "blocks.0.ffn.net_0.proj.weight"
    ]
    assert mapper.apply_list(["blocks.0.ffn.net.2.weight"]) == [
        "blocks.0.ffn.net_2.weight"
    ]
    assert mapper.apply_list(["scale_shift_table"]) == [
        "output_scale_shift_prepare.scale_shift_table"
    ]


def test_wan_adapter_keeps_already_fused_names_stable():
    mapper = Wan22GGUFAdapter.gguf_to_hf_mapper

    assert mapper.apply_list(["blocks.0.attn1.to_qkv.weight"]) == [
        "blocks.0.attn1.to_qkv.weight"
    ]
    assert mapper.apply_list(["blocks.0.ffn.net_0.proj.weight"]) == [
        "blocks.0.ffn.net_0.proj.weight"
    ]
