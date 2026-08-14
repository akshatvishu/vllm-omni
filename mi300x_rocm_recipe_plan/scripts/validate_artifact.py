#!/usr/bin/env python3

import argparse
import glob
import json
import math
from pathlib import Path


def validate_audio(path: Path, expected_sample_rate: int | None, min_rms: float) -> dict:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0:
        raise RuntimeError(f"{path} contains no audio samples")
    if not np.isfinite(audio).all():
        raise RuntimeError(f"{path} contains nonfinite audio samples")
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise RuntimeError(f"{path} sample rate is {sample_rate}, expected {expected_sample_rate}")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    duration = float(audio.shape[0] / sample_rate)
    if duration <= 0:
        raise RuntimeError(f"{path} has a nonpositive duration")
    if not math.isfinite(rms) or rms <= min_rms:
        raise RuntimeError(f"{path} is silent or invalid, RMS={rms}, required>{min_rms}")
    return {
        "path": str(path),
        "kind": "audio",
        "sample_rate": sample_rate,
        "channels": int(audio.shape[1]),
        "samples_per_channel": int(audio.shape[0]),
        "duration_seconds": duration,
        "rms": rms,
        "peak_abs_amplitude": peak,
        "size_bytes": path.stat().st_size,
    }


def validate_image(path: Path, expected_width: int | None, expected_height: int | None) -> dict:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        image_format = image.format
    if expected_width is not None and width != expected_width:
        raise RuntimeError(f"{path} width is {width}, expected {expected_width}")
    if expected_height is not None and height != expected_height:
        raise RuntimeError(f"{path} height is {height}, expected {expected_height}")
    return {
        "path": str(path),
        "kind": "image",
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("audio", "image"))
    parser.add_argument("--glob", required=True)
    parser.add_argument("--expected-sample-rate", type=int)
    parser.add_argument("--min-rms", type=float, default=1e-5)
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    args = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(args.glob))]
    if not paths:
        raise RuntimeError(f"No artifact matched {args.glob}")
    if args.kind == "audio":
        results = [validate_audio(path, args.expected_sample_rate, args.min_rms) for path in paths]
    else:
        results = [validate_image(path, args.expected_width, args.expected_height) for path in paths]
    print(json.dumps({"status": "pass", "artifacts": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
