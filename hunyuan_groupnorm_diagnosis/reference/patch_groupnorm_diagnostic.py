# SPDX-License-Identifier: Apache-2.0

"""Patch ``initialize_model`` to replace VAE GroupNorm with AITER GroupNorm on ROCm."""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

import vllm_omni.diffusion.registry as _registry_mod

logger = init_logger(__name__)

_original_initialize_model = _registry_mod.initialize_model


def _replace_groupnorm_with_aiter(vae: nn.Module) -> bool:
    from aiter.ops.groupnorm import GroupNorm as AiterGroupNorm

    class DiagnosticAiterGroupNorm(AiterGroupNorm):
        def _dump_case(
            self,
            input: torch.Tensor,
            expected: torch.Tensor,
            actual: torch.Tensor | None,
            error: str | None = None,
        ) -> Path:
            output_dir = Path(
                os.environ.get(
                    "HUNYUAN_GROUPNORM_DIAG_DIR",
                    "hunyuan_groupnorm_diagnosis/artifacts/tensor_dumps",
                )
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            dump_path = output_dir / f"groupnorm-mismatch-{os.getpid()}.pt"
            payload = {
                "input": input.detach().cpu(),
                "weight": self.weight.detach().cpu(),
                "bias": self.bias.detach().cpu(),
                "expected": expected.detach().cpu(),
                "num_groups": self.num_groups,
                "eps": self.eps,
                "autocast_enabled": torch.is_autocast_enabled(input.device.type),
            }
            if actual is not None:
                payload["actual"] = actual.detach().cpu()
            if error is not None:
                payload["error"] = error
            torch.save(payload, dump_path)
            return dump_path

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            assert self.weight is not None
            assert self.bias is not None

            print(
                "GROUPNORM_PROBE",
                "shape",
                tuple(input.shape),
                "input",
                input.dtype,
                "weight",
                self.weight.dtype,
                "bias",
                self.bias.dtype,
                "autocast",
                torch.is_autocast_enabled(input.device.type),
                flush=True,
            )

            expected = F.group_norm(
                input,
                self.num_groups,
                self.weight,
                self.bias,
                self.eps,
            )
            pre_call_dump_path = self._dump_case(input, expected, None)
            print("GROUPNORM_PRECALL_DUMP", pre_call_dump_path, flush=True)
            torch.accelerator.synchronize()
            print("GROUPNORM_AITER_CALL", pre_call_dump_path, flush=True)
            try:
                actual = super().forward(input)
                torch.accelerator.synchronize()
            except Exception as exc:
                dump_path = self._dump_case(
                    input,
                    expected,
                    None,
                    error=repr(exc),
                )
                raise RuntimeError(f"AITER GroupNorm raised an exception. dump={dump_path}") from exc

            expected_fp32 = expected.float()
            actual_fp32 = actual.float()
            difference = (actual_fp32 - expected_fp32).abs()
            values_close = torch.allclose(
                actual_fp32,
                expected_fp32,
                rtol=1e-3,
                atol=1e-2,
            )

            if actual.dtype != expected.dtype or not values_close:
                dump_path = self._dump_case(input, expected, actual)
                raise RuntimeError(
                    "AITER GroupNorm mismatch: "
                    f"shape={tuple(input.shape)}, "
                    f"input={input.dtype}, "
                    f"weight={self.weight.dtype}, "
                    f"bias={self.bias.dtype}, "
                    f"expected={expected.dtype}, "
                    f"actual={actual.dtype}, "
                    f"mean_error={difference.mean().item()}, "
                    f"max_error={difference.max().item()}, "
                    f"dump={dump_path}"
                )

            return actual

    targets = [
        (parent, name, child)
        for parent in vae.modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.GroupNorm) and child.affine
    ]

    for parent, name, child in targets:
        new_group_norm = DiagnosticAiterGroupNorm(
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
