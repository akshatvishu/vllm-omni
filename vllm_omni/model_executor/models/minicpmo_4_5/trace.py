# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in diagnostics for MiniCPM-o 4.5 long-form speech generation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

_TRACE_ENV = "MINICPMO45_TRACE"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FNV_OFFSET_BASIS = 14695981039346656037
_FNV_PRIME = 1099511628211
_UINT64_MASK = (1 << 64) - 1


def trace_enabled() -> bool:
    return os.getenv(_TRACE_ENV, "").strip().lower() in _TRUE_VALUES


def update_int_hash(previous: int | None, values: Iterable[int]) -> int:
    """Update a stable 64-bit FNV-1a hash with integer codec IDs."""
    digest = _FNV_OFFSET_BASIS if previous is None else int(previous) & _UINT64_MASK
    for value in values:
        digest ^= int(value) & _UINT64_MASK
        digest = (digest * _FNV_PRIME) & _UINT64_MASK
    return digest


def int_sequence_summary(values: Sequence[int] | torch.Tensor | None) -> dict[str, Any]:
    if values is None:
        return {"count": 0, "sha256": None, "head": [], "tail": []}
    tensor = torch.as_tensor(values, dtype=torch.int64).detach().cpu().reshape(-1).contiguous()
    items = tensor.tolist()
    return {
        "count": len(items),
        "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "head": items[:8],
        "tail": items[-8:],
    }


def tensor_summary(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {"shape": None, "dtype": None, "sha256": None}
    tensor = value.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def waveform_summary(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {
            "samples": 0,
            "sha256": None,
            "minimum": None,
            "maximum": None,
            "rms": None,
            "nonfinite": 0,
        }
    tensor = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
    if tensor.numel() == 0:
        return {
            "samples": 0,
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            "minimum": None,
            "maximum": None,
            "rms": None,
            "nonfinite": 0,
        }
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    return {
        "samples": int(tensor.numel()),
        "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        "minimum": float(finite_values.min().item()) if finite_values.numel() else None,
        "maximum": float(finite_values.max().item()) if finite_values.numel() else None,
        "rms": float(torch.sqrt(torch.mean(finite_values.square())).item()) if finite_values.numel() else None,
        "nonfinite": int((~finite).sum().item()),
    }


def text_summary(value: str | None) -> dict[str, Any]:
    text = value or ""
    encoded = text.encode("utf-8")
    return {
        "chars": len(text),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "head": text[:160],
        "tail": text[-160:],
    }


def trace_event(logger: Any, event: str, **fields: Any) -> None:
    if not trace_enabled():
        return
    payload = {"event": event, **fields}
    logger.info(
        "[MiniCPMO45Trace] %s",
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
    )


def _artifact_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _artifact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_artifact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_artifact_value(item) for item in value)
    return value


def save_trace_artifact(
    event: str,
    request_id: str,
    step: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Save exact replay tensors beside the opt-in MiniCPM trace logs."""
    if not trace_enabled():
        return {"path": None, "error": None}
    trace_dir = Path(os.getenv("MINICPMO45_TRACE_DIR") or os.getenv("TMPDIR") or os.getcwd())
    request_hash = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:12]
    path = trace_dir / f"minicpmo45_{event}_{request_hash}_step{step}.pt"
    try:
        trace_dir.mkdir(parents=True, exist_ok=True)
        torch.save(_artifact_value(payload), path)
    except Exception as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {"path": str(path), "error": None}
