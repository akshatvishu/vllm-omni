# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional experimental MoonMath BF16 attention for measured MI300X shapes."""

import importlib
from functools import cache

import torch
from torch.library import Library
from vllm.utils.torch_utils import direct_register_custom_op


@cache
def _moonmath_forward():
    try:
        return importlib.import_module("moonmath_amd").forward
    except (ImportError, OSError):
        return None


def moonmath_devices() -> tuple[int, ...]:
    """Resolve the optional dependency and supported devices before compilation."""
    if not torch.version.hip or _moonmath_forward() is None:
        return ()
    devices = []
    for index in range(torch.accelerator.device_count()):
        properties = torch.cuda.get_device_properties(index)
        if properties.gcnArchName.split(":")[0] == "gfx942" and properties.multi_processor_count == 304:
            devices.append(index)
    return tuple(devices)


def _moonmath_attention_out_impl(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output: torch.Tensor
) -> None:
    # Keep public V packing and workspace allocation. Never share output between calls.
    _moonmath_forward()(query, key, value, out=output, round_mode="rtna", layout="bshd")


def _moonmath_attention_out_fake(query, key, value, output):
    return None


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "moonmath_attention_out"):
    direct_register_custom_op(
        op_name="moonmath_attention_out",
        op_func=_moonmath_attention_out_impl,
        fake_impl=_moonmath_attention_out_fake,
        mutates_args=["output"],
        target_lib=_OMNI_OP_LIB,
        dispatch_key="CUDA",
    )


def moonmath_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(query)
    torch.ops.vllm_omni.moonmath_attention_out(query, key, value, output)
    return output
