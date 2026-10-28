# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental functional boundary for unchanged AITER inference attention."""
import torch
from aiter.ops.mha import flash_attn_func as native_flash

from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import piecewise_attn

# Populated only by the diagnostic, outside graph capture. Not production state.
AUDIT = None


def _native(q, k, v, *, softmax_scale, causal):
    result = native_flash(q, k, v, softmax_scale=softmax_scale, causal=causal)
    return result[0] if isinstance(result, tuple) else result


def _validate(q, k, v):
    assert torch.version.hip and 'gfx942' in torch.cuda.get_device_properties(q.device).gcnArchName
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert q.device == k.device == v.device
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape[0] == k.shape[0] == v.shape[0] == 1
    assert q.shape[2:] == k.shape[2:] == v.shape[2:] == (32, 128)
    assert k.shape[1] == v.shape[1] and all(x.stride(-1) == 1 for x in (q, k, v))


@torch.library.custom_op('qwen21_functional_probe::segment', mutates_args=())
def segment(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            scale: float, causal: bool) -> torch.Tensor:
    _validate(q, k, v)
    before = [x.clone() for x in (q, k, v)] if AUDIT is not None else None
    # No out= input, dtype change, contiguous conversion, or retained output.
    output = _native(q, k, v, softmax_scale=scale, causal=causal)
    if AUDIT is not None:
        AUDIT['segments'].append(dict(inputs=before, after=[x.clone() for x in (q, k, v)],
            metadata=[(tuple(x.shape), tuple(x.stride()), x.storage_offset()) for x in (q, k, v)],
            output=output.clone(), causal=causal,
            fresh=all(output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr() for x in (q, k, v)),
            stride=tuple(output.stride())))
    return output


@segment.register_fake
def _segment_fake(q, k, v, scale, causal):
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2], v.shape[3]))


def functional_flash(q, k, v, *, softmax_scale, causal=False):
    return segment(q, k, v, softmax_scale, causal)


@torch.library.custom_op('qwen21_functional_probe::reference', mutates_args=())
def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              flat_spans: list[int], scale: float) -> torch.Tensor:
    """Compile the surrounding block; run main's whole attention eagerly here."""
    _validate(q, k, v)
    if flat_spans:
        spans = list(zip(flat_spans[::2], flat_spans[1::2], strict=True))
        output = piecewise_attn(q, k, v, [spans], scale, _native)
    else:
        output = _native(q, k, v, softmax_scale=scale, causal=False)
    if AUDIT is not None:
        AUDIT['reference'] = dict(inputs=[x.clone() for x in (q, k, v)], output=output.clone())
    return output


@reference.register_fake
def _reference_fake(q, k, v, flat_spans, scale):
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2], v.shape[3]))


def reference_forward(self, q, k, v, metadata=None):
    assert metadata is None or (metadata.attn_mask is None and not metadata.extra
                                 and metadata.query_ranges is None)
    spans = [] if metadata is None or metadata.full_attn_spans is None else metadata.full_attn_spans[0]
    return reference(q, k, v, [item for span in spans for item in span], self.softmax_scale)
