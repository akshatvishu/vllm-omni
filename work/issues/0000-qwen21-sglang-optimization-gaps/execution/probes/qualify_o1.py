# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU preflight of checked-out O1 guards and installed HF topology.

Never starts a GPU job or downloads a checkpoint. This cannot qualify real
encoder values, load peak, transfer ordering, or exception recovery. Actual
pipeline evidence must come from activation_stats_offload during diagnostics.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    import torch
    import transformers
    from torch import nn
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    root = Path(a.source_root).resolve()
    source = root / "vllm_omni/diffusion/models/qwen_image_21/pipeline_qwen_image_21.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenImage21Pipeline")
    assignment = next(
        n
        for n in cls.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_offload_plan" for t in n.targets)
    )
    ns = {"OffloadPlan": lambda **kw: kw}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])), str(source), "exec"), ns)
    paths = tuple(ns["_offload_plan"]["encoder_block_attrs"]["text_encoder"])
    expected = ("model.language_model.layers", "model.visual.blocks", "model.visual.deepstack_merger_list")
    assert paths == expected, paths
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 64,
        },
        vision_config={
            "depth": 4,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [1, 3],
        },
    )
    with torch.device("meta"):
        encoder = Qwen3VLForConditionalGeneration(config)
    groups = [encoder.get_submodule(path) for path in paths]
    assert all(isinstance(g, nn.ModuleList) and len(g) >= 2 for g in groups)
    assert len({id(b) for g in groups for b in g}) == sum(map(len, groups))
    constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    guard = next(
        n
        for n in constructor.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "stream_encoder"
    )
    code = compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), str(source), "exec")
    names = (
        "tensor_parallel_size",
        "sequence_parallel_size",
        "data_parallel_size",
        "cfg_parallel_size",
        "pipeline_parallel_size",
    )

    def run(overrides=None, quant=None, components=frozenset({"text_encoder"}), allgather=False, parallel=None):
        c = SimpleNamespace(
            parallel_config=SimpleNamespace(**(dict.fromkeys(names, 1) | (parallel or {}))), **(overrides or {})
        )
        policy = SimpleNamespace(components=components, uses_allgather=lambda _: allgather)
        exec(
            code,
            dict(stream_encoder=True, od_config=c, text_encoder_quant_config=quant, resolve_offload=lambda _: policy),
        )

    run()
    checks = []
    bad = [
        ("custom-loader", dict(overrides={"custom_pipeline_args": {}})),
        ("load-format", dict(overrides={"diffusion_load_format": "diffusers"})),
        ("quantization", dict(quant=object())),
        ("allgather", dict(allgather=True)),
        ("dit-selection", dict(components={"dit", "text_encoder"})),
        ("hsdp", dict(parallel={"use_hsdp": True})),
    ]
    bad += [(name, dict(parallel={name: 2})) for name in names]
    for name, kwargs in bad:
        try:
            run(**kwargs)
        except ValueError as error:
            checks.append({"name": name, "rejected": True, "error": str(error)})
        else:
            raise AssertionError(f"unsupported {name} accepted")
    encoder.model.visual.deepstack_merger_list = nn.ModuleList([encoder.model.visual.deepstack_merger_list[0]])
    assert len(encoder.get_submodule(paths[-1])) < 2
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "status": "CPU_PREFLIGHT_PASS",
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "groups": [(path, len(group)) for path, group in zip(paths, groups)],
                "guards": checks,
                "single_merger_policy": "unsupported by current ring; fixture detected",
                "actual_checkpoint_loaded": False,
                "gpu_executed": False,
                "missing_qualification": [
                    "Real checkpoint encoder embeddings incl vision/deepstack",
                    "Failed-block recovery and next-request parity",
                    "Cancellation and sleep/wake",
                ],
                "scope": (
                    "Actual checked-out source declarations/guard expressions and installed HF meta topology; "
                    "no real-weight or runtime lifecycle claim"
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
