# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Run in the pushed K1 checkout on the GPU; exact bits plus compiled op evidence."""

import json

import torch

from vllm_omni.diffusion.layers.qwen21_kv_pack import eligible, pack

seen = []


def backend(gm, inputs):
    seen.extend(str(n.target) for n in gm.graph.nodes)
    return torch._inductor.compile(gm, inputs)


with torch.inference_mode():
    compiled = torch.compile(pack, backend=backend, fullgraph=True)
    for p, s, h, d in [(0, 1, 2, 128), (17, 33, 4, 128), (4212, 4096, 32, 128)]:
        wide = torch.randn(1, s, 3 * h * d, device="cuda", dtype=torch.bfloat16)
        k = wide[:, :, h * d : 2 * h * d].view(1, s, h, d)
        v = wide[:, :, 2 * h * d :].view(1, s, h, d)
        kp = torch.randn(1, p, h, d, device="cuda", dtype=torch.bfloat16)
        vp = torch.randn_like(kp)
        assert eligible(kp, vp, k, v)
        for fn in (pack, compiled):
            actual = fn(kp, vp, k, v)
            for got, want in zip(actual, (torch.cat((kp, k), 1), torch.cat((vp, v), 1))):
                assert torch.equal(got.view(torch.int16), want.view(torch.int16))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = pack(kp, vp, k, v)
        k.add_(1)
        vp.add_(2)
        graph.replay()
        for got, want in zip(result, (torch.cat((kp, k), 1), torch.cat((vp, v), 1))):
            assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert any("qwen21_kv_pack_experiment" in n for n in seen), seen
    assert not eligible(kp.expand(2, -1, -1, -1), vp.expand(2, -1, -1, -1), k, v)
    for args in (
        (kp.cpu(), vp.cpu(), k.cpu(), v.cpu()),
        (kp.expand(2, -1, -1, -1), vp.expand(2, -1, -1, -1), k.expand(2, -1, -1, -1), v.expand(2, -1, -1, -1)),
        (kp.float(), vp.float(), k.float(), v.float()),
    ):
        try:
            pack(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported direct call was accepted")
print(
    json.dumps(
        {
            "active": True,
            "exact_bits": True,
            "changed_input_graph_replay": True,
            "unsupported_calls_rejected": True,
            "compiled_nodes": seen,
            "scope": "K1 operation only; full-model qualification remains separate",
        }
    )
)
