# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.model_executor.models.utils import WeightsMapper

from .base import GGUFAdapter, gguf_quant_weights_iterator

WAN22_KEYS_RENAME_DICT = {
    "scale_shift_table": "output_scale_shift_prepare.scale_shift_table",
}

WAN22_BLOCK_KEYS_RENAME_DICT = {
    ".attn1.to_q.": ".attn1.to_qkv.",
    ".attn1.to_k.": ".attn1.to_qkv.",
    ".attn1.to_v.": ".attn1.to_qkv.",
    ".attn1.to_out.0.": ".attn1.to_out.",
    ".ffn.net.0.": ".ffn.net_0.",
    ".ffn.net.2.": ".ffn.net_2.",
}


class Wan22GGUFAdapter(GGUFAdapter):
    """GGUF adapter for Wan2.2 transformer checkpoints."""

    @staticmethod
    def is_compatible(od_config, model: torch.nn.Module, source) -> bool:
        model_class = od_config.model_class_name or ""
        if model_class.startswith("Wan"):
            return True
        cfg = od_config.tf_model_config
        if cfg is not None:
            model_type = str(cfg.get("model_type", "")).lower()
            if model_type in {"wan", "wan2_2", "wan2.2"}:
                return True
        return False

    gguf_to_hf_mapper = WeightsMapper(
        orig_to_new_prefix=WAN22_KEYS_RENAME_DICT,
        orig_to_new_substr=WAN22_BLOCK_KEYS_RENAME_DICT,
    )

    def weights_iterator(self) -> Iterable[tuple[str, torch.Tensor]]:
        weights = gguf_quant_weights_iterator(self.gguf_file)
        yield from self.gguf_to_hf_mapper.apply(weights)
