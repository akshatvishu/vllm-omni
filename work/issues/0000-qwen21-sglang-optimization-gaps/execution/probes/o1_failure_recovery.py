# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real-loaded-pipeline diagnostic API; no model construction or GPU auto-run.

Call probe(pipeline, encode, expected, output_path) after actual O1 pipeline
loading. encode() MUST execute actual text+vision encoder on saved real inputs
with the incumbent final-norm capture semantics and return its hidden tensor.
expected is saved B0 encoder output for those inputs. Do not supply frozen
encoder output as encode(). Requires the common transfer recorder for proof.
"""

import json
from pathlib import Path

import torch


class InjectedBlockError(RuntimeError):
    pass


def equal(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and bool(torch.isfinite(a).all())
        and bool(torch.isfinite(b).all())
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def probe(pipeline, encode, expected, output_path):
    encoder = pipeline.text_encoder
    paths = ("model.language_model.layers", "model.visual.blocks", "model.visual.deepstack_merger_list")
    groups = getattr(encoder, "_omni_layerwise_block_groups", [])
    if len(groups) != 3:
        raise ValueError("Actual three-stack offload installation required")
    if not any(p.device.type == "cuda" for p in pipeline.transformer.parameters()):
        raise ValueError("Actual GPU pipeline required")
    rows = []
    with torch.inference_mode():
        initial = encode()
        if not equal(initial, expected.to(initial.device)):
            raise ValueError("Actual encoder does not match saved B0 before failure injection")
        for path in paths:
            block = encoder.get_submodule(path)[0]
            registry = getattr(block, "_hook_registry", None)
            if registry is None or "layerwise_offload" not in registry._hooks:
                raise ValueError("Missing actual offload hook")
            original = block._omni_original_forward
            count = [0]

            def fail(*args, **kwargs):
                count[0] += 1
                # Dispatch already ran prefetch hooks. Raise within original
                # block body so ordinary offload post_forward is truly skipped.
                raise InjectedBlockError(path)

            try:
                block._omni_original_forward = fail
                try:
                    encode()
                except InjectedBlockError:
                    pass
                else:
                    raise AssertionError("Named text/vision/deepstack block never executed")
            finally:
                block._omni_original_forward = original
            resident_after_error = sum(h.is_materialized for h in encoder._omni_layerwise_hooks)
            recovery_error = None
            exact = False
            try:
                recovered = encode()
                exact = equal(recovered, expected.to(recovered.device))
            except Exception as error:
                recovery_error = repr(error)
            rows.append(
                {
                    "stack": path,
                    "injected_calls": count[0],
                    "materialized_blocks_after_error": resident_after_error,
                    "next_request_exact": exact,
                    "recovery_error": recovery_error,
                }
            )
            if recovery_error:
                break
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    report = {
        "status": "PASS"
        if len(rows) == 3 and all(r["next_request_exact"] and r["injected_calls"] for r in rows)
        else "FAIL",
        "rows": rows,
        "scope": (
            "Actual encoder exception and immediate next-call recovery only; "
            "not cancellation/sleep/shutdown qualification"
        ),
        "post_forward_cleanup": (
            "Generic post_forward is skipped by injected error; retained blocks are recorded, "
            "not automatically described as released"
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
