# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/fm/CFM.py


import torch
from torch import nn

from vllm_omni.model_executor.models.common.ming.fm import Solver, build_timesteps


class CFM(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        noise,
        c,
        latent_history,
        steps=10,
        cfg_scale=1.0,
        sway_sampling_coef=-1.0,
        use_epss=True,
        patch_size=1,
        sigma=0.25,
        temperature=1.5,
    ):
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if noise.ndim != 3:
            raise ValueError(f"Expected noise rank-3 [Batch, Dimension, Time], got {tuple(noise.shape)}")
        if c.ndim != 3:
            raise ValueError(f"Expected conditioning rank-3 [Batch, Time, Dimension], got {tuple(c.shape)}")
        if latent_history.ndim != 3:
            raise ValueError(
                f"Expected latent_history rank-3 [Batch, Time, Dimension], got {tuple(latent_history.shape)}"
            )
        if noise.shape[0] != c.shape[0] or noise.shape[0] != latent_history.shape[0]:
            raise ValueError(
                "Batch mismatch across noise, conditioning, and latent_history: "
                f"{noise.shape[0]}, {c.shape[0]}, {latent_history.shape[0]}"
            )
        if noise.shape[-1] != patch_size:
            raise ValueError(f"noise time dim mismatch: got {noise.shape[-1]}, expected patch_size={patch_size}")

        def fn(t, x):
            if cfg_scale < 1e-5:
                if t.ndim == 0:
                    t = t.repeat(x.shape[0])
                pred = self.model(
                    x=x,
                    t=t,
                    c=torch.zeros_like(c),
                    latent_history=latent_history,
                )
                return pred[:, -patch_size:, :]

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.model.forward_with_cfg(
                x=x,
                t=t,
                c=c,
                latent_history=latent_history,
                cfg_scale=cfg_scale,
                patch_size=patch_size,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_scale

        y0 = noise.transpose(1, 2)
        t = build_timesteps(
            steps,
            device=self.device,
            dtype=noise.dtype,
            use_epss=use_epss,
            sway_sampling_coef=sway_sampling_coef,
        )

        solver = Solver(fn, y0, sigma=sigma, temperature=temperature)
        trajectory = solver.integrate(t)
        sampled = trajectory[-1]
        out = sampled

        return out, trajectory
