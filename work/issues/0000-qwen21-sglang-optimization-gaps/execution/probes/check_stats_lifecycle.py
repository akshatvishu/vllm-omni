# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU class lifecycle test. NOT actual pipeline activation evidence."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    source = Path(a.source)
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_VaeLatentStats")
    ns = {"torch": torch, "nn": nn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), ns)
    rows = []
    old = torch.get_default_dtype()
    try:
        for default in (torch.float32, torch.float64):
            torch.set_default_dtype(default)
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                cache = ns["_VaeLatentStats"]()
                cfg = SimpleNamespace(latents_mean=[-0.137, 0.257], latents_std=[0.731, 1.397])
                x = torch.linspace(-2, 2, 48, device="cuda", dtype=dtype).reshape(2, 2, 1, 3, 4)
                mean, std = cache.get(cfg, 2, x)
                bm = torch.tensor(cfg.latents_mean).view(1, 2, 1, 1, 1).to(x.device, x.dtype)
                bs = torch.tensor(cfg.latents_std).view(1, 2, 1, 1, 1).to(x.device, x.dtype)

                def equal(a, b):
                    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

                assert equal((x - mean) / std, (x - bm) / bs) and equal(x * std + mean, x * bs + bm)
                warm = cache.get(cfg, 2, x)
                assert warm[0] is mean and warm[1] is std
                assert not cache.state_dict()
                cfg.latents_mean[0] = 0.789
                cache.get(cfg, 2, x)
                assert cache.mean is not mean
                cache.to(dtype=torch.float64)
                assert cache.mean is None and cache.std is None and cache._key is None
                cache.get(cfg, 2, x)
                cache.clear()
                assert cache.mean is None
                cfg.latents_mean = [0.0]
                try:
                    cache.get(cfg, 2, x)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("malformed config accepted")
                assert cache._key is None and cache.mean is None
                rows.append({"default": str(default), "dtype": str(dtype), "exact": True, "lifecycle": True})
    finally:
        torch.set_default_dtype(old)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(
        json.dumps(
            {
                "scope": "AST class GPU test only; not pipeline activation",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
