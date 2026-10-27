# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Build one comparison group from completed runner rounds, without inferring activation.

Example: --rounds r1 r2 r3 --diagnostic diag --domains images tensors timings
 --tensor-names decode-input.pt vae-decode-output.pt --cases t2i-accuracy
 --device-policy frozen-device-policy.json --output comparison

Use separate outputs per execution mode and controlled/observational workload group.
Images come from the diagnostic run when supplied, otherwise all measured requests.
Tensor names are explicit: do not include unstable fresh encoder captures in a
controlled packing comparison. Historical images use --domains images, not timings.
Candidate adds --baseline-manifest B0/comparison-manifest.json and optionally
--activation evidence.json. Evidence must itself contain active=true/false; no
activation is inferred from successful requests or file presence.
"""

import argparse
import json
import shutil
from pathlib import Path

from provenance import digest, write


def read(path):
    return json.loads(Path(path).read_text())


def normalized(prov, launch, policy):
    command = list(launch["command"])
    # Interpreter and source/checkpoint locations are transport-specific; model content is hashed.
    command[0] = "<python>"
    command[command.index("serve") + 1] = "<checkpoint>"
    if "--diffusion-offload-config" in command:
        index = command.index("--diffusion-offload-config")
        offload_policy = json.loads(command[index + 1])
        if prov["arguments"]["variant"] != "O1" or offload_policy != {"mode": "layer", "components": ["text_encoder"]}:
            raise ValueError("Unexpected offload treatment")
        del command[index : index + 2]
    return {
        "model": prov["checkpoint_files"],
        "inputs": {Path(k).name: v for k, v in prov["inputs"].items()},
        "packages": prov["packages"],
        "harness": prov["harness"],
        "comparable_config": {
            "command": command,
            "prompt_cache": "disabled",
            "mode": prov["arguments"]["mode"],
            "historical": prov["arguments"]["historical"],
        },
        "device_policy": policy,
    }


def build(args):
    rounds = [Path(p).resolve() for p in args.rounds]
    diag = Path(args.diagnostic).resolve() if args.diagnostic else None
    roots = rounds + ([diag] if diag else [])
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("need distinct completed input runs")
    if "timings" in args.domains and len(rounds) != 3:
        raise ValueError("timings require exactly three process rounds")
    if "tensors" in args.domains and (diag is None or not args.tensor_names):
        raise ValueError("tensor comparison requires diagnostic run and explicit tensor names")
    policy = read(args.device_policy)
    if not policy:
        raise ValueError("device policy must be explicit and nonempty")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    canonical = None
    sources = []
    samples = {}
    try:
        for root in roots:
            if read(root / "status.json")["status"] != "completed":
                raise ValueError(f"incomplete run {root}")
            prov = read(root / "provenance.json")
            launch = read(root / "launch.json")
            current = normalized(prov, launch, policy)
            if canonical is None:
                canonical = current
            elif current != canonical:
                raise ValueError(f"mixed modes/configuration/provenance: {root}")
            sources.append(
                {"run": str(root), "sha": prov["sha"], "provenance_sha256": digest(root / "provenance.json")}
            )
            samples[root] = read(root / "samples.json")
            if not all(row.get("success") for row in samples[root]):
                raise ValueError(f"failed request in {root}")
            if root == diag and not prov["arguments"]["diagnostic"]:
                raise ValueError("diagnostic argument is not a diagnostic run")
            if root in rounds and prov["arguments"]["diagnostic"]:
                raise ValueError("diagnostics cannot supply request timings")
        if len({s["sha"] for s in sources}) != 1:
            raise ValueError("mixed source commits within comparison arm")
        cases = args.cases or sorted({row["case"] for row in samples[diag or rounds[0]]})
        if not cases or len(set(cases)) != len(cases):
            raise ValueError("empty/duplicate cases")

        def artifact(source, relative, identity):
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            return {"id": identity, "path": str(relative), "sha256": digest(destination)}

        rows = []
        for case in cases:
            if Path(case).name != case:
                raise ValueError("unsafe case ID")
            row = {
                "id": case,
                "historical": case.startswith("historical-"),
                "observational_edit": case in args.observational_cases,
            }
            if "timings" in args.domains:
                values = []
                for root in rounds:
                    selected = [
                        r["http_seconds"] for r in samples[root] if r["case"] == case and r["phase"] == "measured"
                    ]
                    if len(selected) < 10:
                        raise ValueError(f"{case}: fewer than ten measured samples in {root}")
                    values.append(selected)
                row["timings"] = {"rounds": values, "min_samples": 10}
            if "images" in args.domains:
                entries = []
                image_roots = [diag] if diag else rounds
                for ri, root in enumerate(image_roots):
                    phase = "diagnostic" if root == diag else "measured"
                    chosen = [r for r in samples[root] if r["case"] == case and r["phase"] == phase]
                    if not chosen:
                        raise ValueError(f"missing {phase} image {case} in {root}")
                    for index, sample in enumerate(chosen):
                        source = root / "requests" / sample["id"] / "image.png"
                        if digest(source) != sample["png_sha256"]:
                            raise ValueError("request image hash mismatch")
                        identity = f"round{ri}-sample{index}"
                        entries.append(artifact(source, Path("images") / case / (identity + ".png"), identity))
                row["images"] = entries
            if "tensors" in args.domains:
                entries = []
                for name in args.tensor_names:
                    if Path(name).name != name or not name.endswith(".pt"):
                        raise ValueError("tensor names must be .pt basenames")
                    entries.append(artifact(diag / "captures" / case / name, Path("tensors") / case / name, name))
                row["tensors"] = entries
            rows.append(row)
        manifest = {
            "provenance": canonical,
            "source_runs": sources,
            "required_domains": args.domains,
            "cases": rows,
            "selection": {"tensor_names": args.tensor_names, "image_source": "diagnostic" if diag else "measured"},
            "variant": prov["arguments"]["variant"],
            "treatment": {"mode": "layer", "components": ["text_encoder"]}
            if prov["arguments"]["variant"] == "O1"
            else None,
        }
        if args.baseline_manifest:
            manifest["baseline_manifest_sha256"] = digest(args.baseline_manifest)
        if args.activation:
            evidence = read(args.activation)
            if not isinstance(evidence.get("active"), bool):
                raise ValueError("activation evidence needs explicit boolean active")
            entry = artifact(Path(args.activation), Path("activation.json"), "activation")
            manifest["activation"] = {"active": evidence["active"], "path": entry["path"], "sha256": entry["sha256"]}
        write(output / "comparison-manifest.json", manifest)
    except Exception as exc:
        write(output / "aggregate-error.json", {"error": repr(exc)})
        raise
    return output / "comparison-manifest.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", nargs="*", default=[])
    p.add_argument("--diagnostic")
    p.add_argument("--domains", nargs="+", choices=["images", "tensors", "timings"], required=True)
    p.add_argument("--tensor-names", nargs="*", default=[])
    p.add_argument("--cases", nargs="*")
    p.add_argument("--observational-cases", nargs="*", default=[])
    p.add_argument("--device-policy", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline-manifest")
    p.add_argument("--activation")
    print(build(p.parse_args()))


if __name__ == "__main__":
    main()
