# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU owner/layout probe for P1 checkout; GPU model replay remains mandatory."""

import torch

from vllm_omni.diffusion.models.qwen_image_21.decode_graph import QwenImage21DecodeGraphEntry

entry = QwenImage21DecodeGraphEntry(
    branch="cond",
    batch_size=1,
    prefix_len=3,
    target_tokens=5,
    in_channels=4,
    dtype=torch.bfloat16,
    device=torch.device("cpu"),
    kv_shapes=[torch.Size([1, 3, 2, 4])],
    kv_dtype=torch.bfloat16,
    freqs=torch.ones(5, 2, dtype=torch.complex64),
    retain_attention_buffers=True,
)
source = [
    {
        "cond": {
            "key": torch.ones(1, 3, 2, 4, dtype=torch.bfloat16),
            "value": torch.full((1, 3, 2, 4), 2, dtype=torch.bfloat16),
        }
    }
]
entry.refresh_prefix(source)
assert entry.k[0].untyped_storage().data_ptr() == entry.packed_k[0].untyped_storage().data_ptr()
for value in (3, 7):
    entry.packed_k[0][:, 3:].fill_(value)
    entry.packed_v[0][:, 3:].fill_(value + 1)
    entry.refresh_prefix(source)
    assert torch.equal(entry.k[0], source[0]["cond"]["key"])
    assert torch.equal(entry.v[0], source[0]["cond"]["value"])
    assert bool((entry.packed_k[0][:, 3:] == value).all())
assert "packed_key" not in source[0]["cond"]
print("P1 static owner probe passed; model compiler/replay/numerics/lifecycle unqualified")
