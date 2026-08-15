# SPDX-License-Identifier: Apache-2.0

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--bad",
        type=Path,
        default=Path("hunyuan_groupnorm_diagnosis/artifacts/bad/images/output_0_0.png"),
    )
    parser.add_argument(
        "--fixed",
        type=Path,
        default=Path("hunyuan_groupnorm_diagnosis/artifacts/fixed/images/output_0_0.png"),
    )
    return parser.parse_args()


def structural_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    channels = first.shape[1]
    coordinates = torch.arange(-5, 6, dtype=first.dtype)
    gaussian = torch.exp(-0.5 * (coordinates / 1.5).square())
    gaussian /= gaussian.sum()
    kernel = torch.outer(gaussian, gaussian).expand(channels, 1, 11, 11)

    first = F.pad(first, (5, 5, 5, 5), mode="reflect")
    second = F.pad(second, (5, 5, 5, 5), mode="reflect")
    inputs = torch.cat((first, second, first.square(), second.square(), first * second))
    means = F.conv2d(inputs, kernel, groups=channels).split(first.shape[0])

    first_mean_squared = means[0].square()
    second_mean_squared = means[1].square()
    mean_product = means[0] * means[1]
    first_variance = torch.clamp(means[2] - first_mean_squared, min=0.0)
    second_variance = torch.clamp(means[3] - second_mean_squared, min=0.0)
    covariance = means[4] - mean_product
    score = ((2 * mean_product + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (first_mean_squared + second_mean_squared + 0.01**2) * (first_variance + second_variance + 0.03**2)
    )
    return float(score.mean().item())


def metrics(first: Image.Image, second: Image.Image) -> dict[str, float]:
    first_array = np.asarray(first.convert("RGB"), dtype=np.float32) / 255.0
    second_array = np.asarray(second.convert("RGB"), dtype=np.float32) / 255.0
    error = np.abs(first_array - second_array)
    first_tensor = torch.from_numpy(first_array).permute(2, 0, 1).unsqueeze(0)
    second_tensor = torch.from_numpy(second_array).permute(2, 0, 1).unsqueeze(0)
    mean_squared_error = float(torch.mean((first_tensor - second_tensor).square()).item())
    psnr = math.inf if mean_squared_error == 0 else -10 * math.log10(mean_squared_error)
    return {
        "mean": float(error.mean()),
        "p99": float(np.quantile(error, 0.99)),
        "ssim": structural_similarity(first_tensor, second_tensor),
        "psnr": psnr,
    }


def print_metrics(label: str, values: dict[str, float]) -> None:
    print(
        label,
        "mean",
        values["mean"],
        "p99",
        values["p99"],
        "ssim",
        values["ssim"],
        "psnr",
        values["psnr"],
    )


def passes_thresholds(values: dict[str, float]) -> bool:
    return values["mean"] <= 0.03 and values["p99"] <= 0.3 and values["ssim"] >= 0.97 and values["psnr"] >= 30


def main() -> None:
    args = parse_args()
    baseline = Image.open(args.baseline).convert("RGB")
    bad = Image.open(args.bad).convert("RGB") if args.bad.is_file() else None
    fixed = Image.open(args.fixed).convert("RGB") if args.fixed.is_file() else None

    bad_metrics = metrics(bad, baseline) if bad is not None else None
    fixed_metrics = metrics(fixed, baseline) if fixed is not None else None

    if bad_metrics is not None:
        print_metrics("bad_vs_baseline", bad_metrics)
        print("bad_passes", passes_thresholds(bad_metrics))
    else:
        print("bad_image_missing", args.bad)

    if fixed_metrics is not None:
        print_metrics("fixed_vs_baseline", fixed_metrics)
        print("fixed_passes", passes_thresholds(fixed_metrics))
    else:
        print("fixed_image_missing", args.fixed)

    if bad is not None and fixed is not None:
        print_metrics("bad_vs_fixed", metrics(bad, fixed))

    print(
        "repository_thresholds",
        "mean<=0.03",
        "p99<=0.3",
        "ssim>=0.97",
        "psnr>=30",
    )


if __name__ == "__main__":
    main()
