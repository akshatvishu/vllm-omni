# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experiment: preserve AITER arithmetic, write prefill segments in place.

The operation is functional. Its output is newly allocated on every invocation.
This prototype intentionally depends on the pinned AITER private forward API.
It is not a generally supported backend or a production default.
"""

import torch

from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import build_segments


@torch.library.custom_op("qwen21_prefill_probe::direct_output", mutates_args=())
def direct_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    flat_spans: list[int],
    scale: float,
) -> torch.Tensor:
    from aiter.ops.mha import _flash_attn_forward

    assert torch.version.hip and "gfx942" in torch.cuda.get_device_properties(query.device).gcnArchName
    assert query.shape[0] == key.shape[0] == value.shape[0] == 1
    assert query.shape[2:] == key.shape[2:] == value.shape[2:] == (32, 128)
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert query.device == key.device == value.device
    assert query.shape[1] == key.shape[1] == value.shape[1]
    assert all(x.stride(-1) == 1 for x in (query, key, value))
    assert len(flat_spans) % 2 == 0
    spans = list(zip(flat_spans[::2], flat_spans[1::2], strict=True))
    end = 0
    for start, stop in spans:
        assert end <= start < stop <= query.shape[1]
        end = stop
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    for segment in build_segments(spans, 0, query.shape[1]):
        target = output[:, segment.q_start : segment.q_end]
        actual = _flash_attn_forward(
            query[:, segment.q_start : segment.q_end],
            key[:, : segment.kv_end],
            value[:, : segment.kv_end],
            dropout_p=0.0,
            softmax_scale=scale,
            causal=segment.mode == "causal",
            window_size_left=-1,
            window_size_right=-1,
            sink_size=0,
            bias=None,
            alibi_slopes=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            return_lse=False,
            return_softmax=False,
            how_v3_bf16_cvt=1,
            out=target,
        )[0]
        assert actual.data_ptr() == target.data_ptr()
    return output


@direct_output.register_fake
def _fake(query, key, value, flat_spans, scale):
    return query.new_empty(query.shape)
