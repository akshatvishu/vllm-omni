# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare immutable B0 artifacts with one candidate; never recollect B0.

Each directory contains comparison-manifest.json:
 {"provenance": {"model":..., "inputs":..., "packages":..., "harness":...,
                  "comparable_config":..., "device_policy":...},
  "required_domains": ["images", "tensors", "timings"],
  "cases": [{"id":"...", "observational_edit":false, "historical":false,
     "images":[{"id":"sample0","path":"...png","sha256":"..."}],
     "tensors":[{"id":"preclamp","path":"...pt","sha256":"..."}],
     "timings":{"rounds":[[1.0,...],[1.0,...],[1.0,...]],"min_samples":10}}]}
Candidate additionally supplies baseline_manifest_sha256 and activation:
 {"active":true,"path":"activation.json","sha256":"..."}.
Tensor artifacts must be nonempty dicts of tensors/metadata, loaded weights_only.
Each timing domain must already exclude warmups and contain three fresh-process
rounds. Source commit differs intentionally and is recorded outside comparable
provenance. Mode/workload/backend/config and input hashes must match exactly.
An observational edit remains XFAIL, never a lossless pass. Historical image
thresholds are reported separately from exact pixel equality. Missing evidence
is ERROR. --self-test performs CPU-only synthetic comparator checks.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

COMPARABLE_KEYS = ("model", "inputs", "packages", "harness", "comparable_config", "device_policy")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(root, entry):
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes run directory")
    if not entry.get("sha256") or digest(path) != entry["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def indexed(items):
    out = {x["id"]: x for x in items}
    if len(out) != len(items) or not out:
        raise ValueError("empty or duplicate artifact/case IDs")
    return out


def image_metrics(a, b):
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity

    with Image.open(a) as im:
        left = np.asarray(im.convert("RGBA")).copy()
    with Image.open(b) as im:
        right = np.asarray(im.convert("RGBA")).copy()
    if left.shape != right.shape:
        raise ValueError("image shapes differ")
    result = {}
    for name, n in [("rgb", 3), ("rgba", 4)]:
        x, y = left[..., :n], right[..., :n]
        mse = float(np.mean((x.astype("float64") - y.astype("float64")) ** 2))
        size = min(7, x.shape[0], x.shape[1])
        size -= 1 - size % 2
        if size < 3:
            raise ValueError("image too small for SSIM")
        result[name] = {
            "exact": bool(np.array_equal(x, y)),
            "baseline_pixel_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "candidate_pixel_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "psnr_db": None if mse == 0 else float(10 * math.log10(255**2 / mse)),
            "psnr_infinite": mse == 0,
            "ssim": float(structural_similarity(x, y, data_range=255, channel_axis=-1, win_size=size)),
        }
    result["historical_pass"] = all(
        v["ssim"] >= 0.99 and (v["psnr_infinite"] or v["psnr_db"] >= 30) for v in (result["rgb"],)
    )
    return result


def tensor_metrics(a, b):
    import torch

    x = torch.load(a, map_location="cpu", weights_only=True)
    y = torch.load(b, map_location="cpu", weights_only=True)
    if not isinstance(x, dict) or not isinstance(y, dict) or not x or x.keys() != y.keys():
        raise ValueError("tensor artifact must have matching nonempty named dictionaries")
    result = {}

    def visit(u, v, key):
        if isinstance(u, torch.Tensor):
            if not isinstance(v, torch.Tensor) or u.shape != v.shape or u.dtype != v.dtype:
                raise ValueError(f"tensor metadata differs: {key}")
            finite = bool(torch.isfinite(u).all() and torch.isfinite(v).all())
            exact = torch.equal(
                u.contiguous().reshape(-1).view(torch.uint8), v.contiguous().reshape(-1).view(torch.uint8)
            )
            result[key] = {
                "exact": exact,
                "finite": finite,
                "shape": list(u.shape),
                "dtype": str(u.dtype),
                "baseline_stride": list(u.stride()),
                "candidate_stride": list(v.stride()),
            }
        elif isinstance(u, dict):
            if not isinstance(v, dict) or u.keys() != v.keys():
                raise ValueError(f"metadata keys differ: {key}")
            for k, value in u.items():
                if not isinstance(k, str):
                    raise ValueError("metadata keys must be strings")
                visit(value, v[k], f"{key}.{k}" if key else k)
        elif isinstance(u, (list, tuple)):
            if type(v) is not type(u) or len(u) != len(v):
                raise ValueError(f"metadata sequence differs: {key}")
            for i, (a, b) in enumerate(zip(u, v)):
                visit(a, b, f"{key}[{i}]")
        else:
            if type(u) is not type(v) or u != v:
                raise ValueError(f"metadata differs: {key}")

    visit(x, y, "")
    if not result:
        raise ValueError("tensor artifact contains no tensors")
    return result


def timing_metrics(a, b):
    required = max(a.get("min_samples", 10), b.get("min_samples", 10), 10)

    def medians(value):
        rounds = value["rounds"]
        if len(rounds) != 3 or any(len(r) < required for r in rounds):
            raise ValueError("timing needs three complete process rounds")
        if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for r in rounds for t in r):
            raise ValueError("invalid timing sample")
        return [statistics.median(r) for r in rounds]

    x, y = medians(a), medians(b)
    noise = max(x) - min(x)
    gain = statistics.median(x) - statistics.median(y)
    return {
        "baseline_round_medians": x,
        "candidate_round_medians": y,
        "baseline_range": noise,
        "median_improvement": gain,
        "demonstrated_speed_gain": all(t < min(x) for t in y) and gain > noise,
    }


def compare(baseline, candidate):
    bpath = baseline / "comparison-manifest.json"
    cpath = candidate / "comparison-manifest.json"
    b = json.loads(bpath.read_text())
    c = json.loads(cpath.read_text())
    if c.get("baseline_manifest_sha256") != digest(bpath):
        raise ValueError("candidate does not reference this sealed B0 manifest")
    if b.get("treatment") is not None:
        raise ValueError("B0 must use the resident baseline configuration")
    expected_treatment = {"mode": "layer", "components": ["text_encoder"]} if c.get("variant") == "O1" else None
    if c.get("treatment") != expected_treatment:
        raise ValueError("Unexpected optimization-specific configuration")
    for key in COMPARABLE_KEYS:
        if key not in b.get("provenance", {}) or key not in c.get("provenance", {}):
            raise ValueError(f"missing provenance {key}")
        if b["provenance"][key] != c["provenance"][key]:
            raise ValueError(f"incomparable provenance {key}")
    act = c.get("activation", {})
    activation_error = None
    try:
        if act.get("active") is not True:
            raise ValueError("candidate activation not demonstrated")
        artifact(candidate, act)
    except (ValueError, KeyError, OSError) as exc:
        activation_error = str(exc)
    domains = b.get("required_domains")
    if not domains or set(domains) - {"images", "tensors", "timings"} or c.get("required_domains") != domains:
        raise ValueError("missing or inconsistent required domains")
    bs, cs = indexed(b["cases"]), indexed(c["cases"])
    if bs.keys() != cs.keys():
        raise ValueError("case IDs differ")
    results = {}
    failed = False
    observational = False
    for name, left in bs.items():
        right = cs[name]
        row = {}
        exact = True
        tensors_exact = True
        for domain in domains:
            if domain == "timings":
                row[domain] = timing_metrics(left[domain], right[domain])
                continue
            ls, rs = indexed(left[domain]), indexed(right[domain])
            if ls.keys() != rs.keys():
                raise ValueError(f"{name}: {domain} IDs differ")
            row[domain] = {}
            for ident, left_artifact in ls.items():
                a, bp = artifact(baseline, left_artifact), artifact(candidate, rs[ident])
                value = image_metrics(a, bp) if domain == "images" else tensor_metrics(a, bp)
                row[domain][ident] = value
                matched = (
                    value["rgba"]["exact"]
                    if domain == "images"
                    else all(v["exact"] and v["finite"] for v in value.values())
                )
                exact &= matched
                if domain == "tensors":
                    tensors_exact &= matched
        if left.get("observational_edit", False):
            row["status"] = "XFAIL_OBSERVATIONAL" if tensors_exact else "FAIL"
            observational = True
            failed |= not tensors_exact
        else:
            row["status"] = "PASS" if exact else "FAIL"
            failed |= not exact
        row["lossless_exact"] = bool(exact)
        if left.get("historical", False):
            row["historical_pass"] = bool(row.get("images")) and all(
                v["historical_pass"] for v in row["images"].values()
            )
        results[name] = row
    return {
        "status": "FAIL"
        if failed
        else ("UNQUALIFIED" if activation_error else ("INCOMPLETE_OBSERVATIONAL" if observational else "PASS")),
        "activation_error": activation_error,
        "baseline_manifest_sha256": digest(bpath),
        "candidate_manifest_sha256": digest(cpath),
        "cases": results,
    }


def self_test():
    import tempfile

    import numpy as np
    import torch
    from PIL import Image

    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory)
        x = np.full((16, 16, 4), 128, dtype=np.uint8)
        y = x.copy()
        y[0, 0, 3] = 0
        Image.fromarray(x).save(p / "a.png")
        Image.fromarray(y).save(p / "b.png")
        m = image_metrics(p / "a.png", p / "b.png")
        assert m["rgb"]["exact"] and not m["rgba"]["exact"]
        torch.save({"preclamp": torch.tensor([0.0])}, p / "a.pt")
        torch.save({"preclamp": torch.tensor([-0.0])}, p / "b.pt")
        assert not tensor_metrics(p / "a.pt", p / "b.pt")["preclamp"]["exact"]
        b = {"rounds": [[t] * 10 for t in (10, 11, 12)]}
        c = {"rounds": [[9] * 10] * 3}
        assert not timing_metrics(b, c)["demonstrated_speed_gain"]
        c = {"rounds": [[7] * 10] * 3}
        assert timing_metrics(b, c)["demonstrated_speed_gain"]
        try:
            timing_metrics(b, {"rounds": [[7] * 10] * 2})
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete timing accepted")
    print("PASS: alpha-only difference, signed-zero tensor difference, saved-B0 noise rule, missing timing round")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not all((args.baseline, args.candidate, args.output)):
            parser.error("--baseline --candidate --output required")
        try:
            report = compare(args.baseline, args.candidate)
        except Exception as exc:
            report = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise SystemExit(0 if report["status"] == "PASS" else 1)
