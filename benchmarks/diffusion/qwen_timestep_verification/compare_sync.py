# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare correctness directories or aggregate fresh-process request timings."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.baseline) == len(args.candidate)
    result = {"pairs": [], "complete": False}
    for baseline, candidate in zip(args.baseline, args.candidate):
        left = json.loads((baseline / "report.json").read_text())
        right = json.loads((candidate / "report.json").read_text())
        assert left["complete"] and right["complete"]
        for key in ("model", "phase", "steps", "size", "prompts", "eager", "repeats", "warmup"):
            assert left["args"][key] == right["args"][key], key
        assert len(left["requests"]) == len(right["requests"])
        pair = {"baseline": str(baseline), "candidate": str(candidate), "requests": []}
        for lreq, rreq in zip(left["requests"], right["requests"]):
            assert (lreq["prompt"], lreq["seed"]) == (rreq["prompt"], rreq["seed"])
            entry = {"index": lreq["index"], "pixels_equal": lreq["pixel_sha256"] == rreq["pixel_sha256"]}
            if left["args"]["phase"] == "correctness":
                import torch
                la = torch.load(baseline / "audit" / f'request-{lreq["index"]:03d}.pt', weights_only=True)
                ra = torch.load(candidate / "audit" / f'request-{rreq["index"]:03d}.pt', weights_only=True)
                entry["latents_equal"] = torch.equal(la["latents"], ra["latents"])
                entry["latent_max_abs"] = (la["latents"].float()-ra["latents"].float()).abs().max().item()
                entry["build_input_devices"] = {"baseline": la["build_input_devices"], "candidate": ra["build_input_devices"]}
                for name in ("model_timesteps", "scheduler_timesteps"):
                    assert len(la[name]) == len(ra[name]) == left["args"]["steps"]
                    entry[name + "_equal"] = all(torch.equal(l[2], r[2]) and l[1] == r[1]
                        for l, r in zip(la[name], ra[name]))
                assert all(x[0] == "cpu" for x in la["scheduler_timesteps"] + ra["scheduler_timesteps"])
                entry["pass"] = all(entry[k] for k in ("pixels_equal", "latents_equal", "model_timesteps_equal", "scheduler_timesteps_equal"))
            else:
                entry["pass"] = entry["pixels_equal"]
            pair["requests"].append(entry)
        left_ms = [r["request_ms"] for r in left["requests"] if r["measured"]]
        right_ms = [r["request_ms"] for r in right["requests"] if r["measured"]]
        if left_ms:
            pair["timing"] = {"baseline_ms": left_ms, "candidate_ms": right_ms,
                "baseline_median_ms": statistics.median(left_ms), "candidate_median_ms": statistics.median(right_ms),
                "candidate_over_baseline": statistics.median(right_ms) / statistics.median(left_ms)}
        pair["pass"] = all(r["pass"] for r in pair["requests"])
        result["pairs"].append(pair)
    result["complete"] = True
    result["pass"] = all(p["pass"] for p in result["pairs"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
