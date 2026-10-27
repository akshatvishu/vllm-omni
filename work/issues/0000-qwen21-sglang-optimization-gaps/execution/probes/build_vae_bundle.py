# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Build actual decoder-call replay bundle from completed B0 diagnostics.

Uses post_quant_conv decoder inputs and PRECLAMP decoder outputs, never stitched
VAE outputs. Rejects stateful non-first chunks because a standalone replay lacks
preceding feature-cache state. Does not claim full-image/tile-blend qualification.
"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, help="Completed runner diagnostic run directory")
    p.add_argument("--output", required=True, help="New .pt bundle; sidecar .json records provenance")
    p.add_argument("--cases", nargs="*")
    args = p.parse_args()
    import torch

    root = Path(args.baseline).resolve()
    output = Path(args.output).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError(output)
    if json.loads((root / "status.json").read_text()).get("status") != "completed":
        raise ValueError("incomplete baseline")
    provenance = json.loads((root / "provenance.json").read_text())
    if not provenance["arguments"]["diagnostic"]:
        raise ValueError("baseline is not diagnostic")
    names = args.cases or sorted(x.name for x in (root / "captures").iterdir() if x.is_dir())
    rows = []
    evidence = []
    for name in names:
        if Path(name).name != name:
            raise ValueError("invalid case name")
        directory = root / "captures" / name
        metadata = json.loads((directory / "metadata.json").read_text())
        count = metadata["decoder_calls"]
        if count < 1:
            raise ValueError(f"no decoder calls: {name}")
        for index in range(count):
            inp = directory / f"decoder-call-{index:03d}-input.pt"
            out = directory / f"decoder-call-{index:03d}-output.pt"
            x = torch.load(inp, map_location="cpu", weights_only=True)
            y = torch.load(out, map_location="cpu", weights_only=True)
            if x.get("first_chunk") is not True:
                raise ValueError(f"{inp}: stateful chunk cannot be replayed without preceding state")
            if not isinstance(x.get("input"), torch.Tensor) or not isinstance(y.get("output"), torch.Tensor):
                raise TypeError("decoder captures need tensors")
            if not torch.isfinite(x["input"]).all() or not torch.isfinite(y["output"]).all():
                raise ValueError("nonfinite baseline")
            rows.append(
                {
                    "name": f"{name}/decoder-{index:03d}",
                    "input": x["input"],
                    "input_kind": "decoder",
                    "expected_preclamp": y["output"],
                    "first_chunk": True,
                }
            )
            evidence.append(
                {"input": str(inp), "input_sha256": digest(inp), "output": str(out), "output_sha256": digest(out)}
            )
    if len(rows) < 2:
        raise ValueError("need at least two distinct actual decoder inputs")
    # The replay script checks its first two cases; place a genuinely distinct input second.
    different = next((i for i in range(1, len(rows)) if not torch.equal(rows[0]["input"], rows[i]["input"])), None)
    if different is None:
        raise ValueError("all captured inputs are identical")
    rows[1], rows[different] = rows[different], rows[1]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cases": rows}, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "baseline": str(root),
                "baseline_provenance_sha256": digest(root / "provenance.json"),
                "bundle_sha256": digest(output),
                "case_order": [r["name"] for r in rows],
                "source_artifacts": evidence,
                "scope": "Actual decoder-call preclamp parity only; no timed serving or tiled-blend claim.",
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
