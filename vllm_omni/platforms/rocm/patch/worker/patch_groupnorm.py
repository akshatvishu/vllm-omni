# SPDX-License-Identifier: Apache-2.0

"""Patch ``initialize_model`` to replace VAE GroupNorm with AITER GroupNorm on ROCm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

import vllm_omni.diffusion.registry as _registry_mod

logger = init_logger(__name__)

_original_initialize_model = _registry_mod.initialize_model


class _AiterGroupNormAutocastMixin:
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert self.weight is not None and self.bias is not None
        device_type = input.device.type
        if torch.is_autocast_enabled(device_type):
            # PyTorch registers GroupNorm as an FP32 autocast op. AITER bypasses
            # that dispatcher rule, so preserve it before entering the HIP kernel.
            input = input.float()
            weight = self.weight.float()
            bias = self.bias.float()
        else:
            weight = self.weight
            bias = self.bias

        with torch.amp.autocast(device_type, enabled=False):
            if input.dtype != self.weight.dtype or input.dtype != self.bias.dtype:
                output = F.group_norm(input, self.num_groups, weight, bias, self.eps)
                logger.info_once(
                    "PyTorch GroupNorm fallback completed successfully for input dtype %s, "
                    "weight dtype %s, and bias dtype %s.",
                    input.dtype,
                    self.weight.dtype,
                    self.bias.dtype,
                )
                return output

            output = super().forward(input)
            logger.info_once(
                "AITER GroupNorm kernel completed successfully with input dtype %s.",
                input.dtype,
            )
            return output


def _replace_groupnorm_with_aiter(vae: nn.Module) -> bool:
    from aiter.ops.groupnorm import GroupNorm as AiterGroupNorm

    class AutocastAwareAiterGroupNorm(_AiterGroupNormAutocastMixin, AiterGroupNorm):
        pass

    targets = [
        (parent, name, child)
        for parent in vae.modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.GroupNorm) and child.affine
    ]

    for parent, name, child in targets:
        new_group_norm = AutocastAwareAiterGroupNorm(
            num_groups=child.num_groups,
            num_channels=child.num_channels,
            eps=child.eps,
            affine=True,
            device=child.weight.device,
            dtype=child.weight.dtype,
        )
        new_group_norm.weight = child.weight
        new_group_norm.bias = child.bias
        setattr(parent, name, new_group_norm)

    return len(targets) > 0


def _patched_initialize_model(od_config):
    model = _original_initialize_model(od_config)

    if hasattr(model, "vae"):
        try:
            from vllm._aiter_ops import is_aiter_found_and_supported

            if is_aiter_found_and_supported() and _replace_groupnorm_with_aiter(model.vae):
                logger.info("AITER GroupNorm is enabled for VAE.")
        except Exception:
            logger.warning("Failed to apply AITER GroupNorm to VAE.")

    return model


_registry_mod.initialize_model = _patched_initialize_model
