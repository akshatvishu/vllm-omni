# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors


class MingQwen2Backbone(nn.Module):
    """Thin Ming wrapper around upstream vLLM Qwen2Model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model = Qwen2Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

    def get_input_embeddings(self) -> nn.Module:
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        raise AttributeError("Could not locate token embeddings on Ming Qwen2 backbone.")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            return inputs_embeds
        if hasattr(self.model, "embed_input_ids"):
            return self.model.embed_input_ids(input_ids)
        return self.get_input_embeddings()(input_ids)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states)

    def sample(self, logits: torch.Tensor, sampling_metadata: Any):
        return self.model.sample(logits, sampling_metadata)
