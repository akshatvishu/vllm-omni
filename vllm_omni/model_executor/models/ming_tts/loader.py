# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Any

import torch


def load_weights(model_stage: str, model: Any, weights: list[tuple[str, torch.Tensor]]):
    if model_stage == "llm":
        allowed = ("model.", "linear_proj_audio.", "flowloss.", "stop_head.", "spk_head.")
        llm_weights = [(k, v) for k, v in weights if k.startswith(allowed)]
        if not llm_weights:
            raise RuntimeError(
                "Ming Stage-0 received no loadable checkpoint weights. "
                "Expected prefixes: model.*, linear_proj_audio.*, flowloss.*, stop_head.*, spk_head.*"
            )
        loaded = model.load_weights(llm_weights)
        return {f"model.{name}" for name in loaded}

    audio_weights = [(k, v) for k, v in weights if k.startswith("audio.")]
    if not audio_weights:
        raise RuntimeError("Ming Stage-1 received no loadable checkpoint weights. Expected prefix: audio.*")
    loaded = model.load_weights(audio_weights)
    return {f"model.{name}" for name in loaded}
