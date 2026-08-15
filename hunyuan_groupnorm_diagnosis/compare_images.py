# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    "hunyuan_groupnorm_diagnosis/artifacts/matplotlib_cache",
)

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


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


def metrics(first: Image.Image, second: Image.Image) -> dict[str, float]:
    first_array = np.asarray(first.convert("RGB"), dtype=np.float32) / 255.0
    second_array = np.asarray(second.convert("RGB"), dtype=np.float32) / 255.0
    error = np.abs(first_array - second_array)
    first_tensor = torch.from_numpy(first_array).permute(2, 0, 1).unsqueeze(0)
    second_tensor = torch.from_numpy(second_array).permute(2, 0, 1).unsqueeze(0)
    ssim = float(
        StructuralSimilarityIndexMeasure(data_range=1.0)(
            first_tensor,
            second_tensor,
        ).item()
    )
    psnr = float(
        PeakSignalNoiseRatio(data_range=1.0)(
            first_tensor,
            second_tensor,
        ).item()
    )
    return {
        "mean": float(error.mean()),
        "p99": float(np.quantile(error, 0.99)),
        "ssim": ssim,
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


def main() -> None:
    args = parse_args()
    baseline = Image.open(args.baseline).convert("RGB")
    bad = Image.open(args.bad).convert("RGB")
    fixed = Image.open(args.fixed).convert("RGB")

    bad_metrics = metrics(bad, baseline)
    fixed_metrics = metrics(fixed, baseline)
    revision_metrics = metrics(bad, fixed)

    print_metrics("bad_vs_baseline", bad_metrics)
    print_metrics("fixed_vs_baseline", fixed_metrics)
    print_metrics("bad_vs_fixed", revision_metrics)
    print(
        "repository_thresholds",
        "mean<=0.03",
        "p99<=0.3",
        "ssim>=0.97",
        "psnr>=30",
    )
    print(
        "bad_passes",
        bad_metrics["mean"] <= 0.03
        and bad_metrics["p99"] <= 0.3
        and bad_metrics["ssim"] >= 0.97
        and bad_metrics["psnr"] >= 30,
    )
    print(
        "fixed_passes",
        fixed_metrics["mean"] <= 0.03
        and fixed_metrics["p99"] <= 0.3
        and fixed_metrics["ssim"] >= 0.97
        and fixed_metrics["psnr"] >= 30,
    )


if __name__ == "__main__":
    main()
