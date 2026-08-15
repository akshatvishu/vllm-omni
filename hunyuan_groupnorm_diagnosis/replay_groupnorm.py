# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from aiter.ops.groupnorm import groupnorm_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_path = args.dump
    if dump_path.is_dir():
        dumps = sorted(dump_path.glob("groupnorm-mismatch-*.pt"))
        if not dumps:
            raise FileNotFoundError(f"No GroupNorm dumps found in {dump_path}")
        dump_path = dumps[0]
    case = torch.load(dump_path, weights_only=False)

    input = case["input"].cuda()
    weight = case["weight"].cuda()
    bias = case["bias"].cuda()
    num_groups = case["num_groups"]
    eps = case["eps"]

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        expected = F.group_norm(input, num_groups, weight, bias, eps)
        actual = groupnorm_run(input, num_groups, weight, bias, eps)
    torch.accelerator.synchronize()

    error = (actual.float() - expected.float()).abs()
    print("dump", dump_path)
    print("shape", tuple(input.shape))
    print("input", input.dtype)
    print("weight", weight.dtype)
    print("bias", bias.dtype)
    print("expected", expected.dtype)
    print("actual", actual.dtype)
    print("expected_finite", torch.isfinite(expected).all().item())
    print("actual_finite", torch.isfinite(actual).all().item())
    print("mean_error", error.mean().item())
    print("max_error", error.max().item())
    print(
        "close",
        torch.allclose(
            actual.float(),
            expected.float(),
            rtol=1e-3,
            atol=1e-2,
        ),
    )


if __name__ == "__main__":
    main()
