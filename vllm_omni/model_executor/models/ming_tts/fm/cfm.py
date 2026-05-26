# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/fm/CFM.py


import torch
from torch import nn

from vllm_omni.model_executor.models.ming_utils.dit import get_epss_timesteps


class Solver:
    def __init__(self, func, y0, sigma=0.25, temperature=1.5) -> None:
        self.func = func
        self.y0 = y0
        self.sigma = sigma
        self.temperature = temperature

    def integrate(self, t):
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            f0 = self.func(t0, y0)
            dy = dt * f0
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1

            noise = torch.randn_like(y0)
            shift = self.sigma * (self.temperature**0.5) * (abs(dt) ** 0.5) * noise
            y0 = y1 + shift

        return solution

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


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
        if use_epss:
            t = get_epss_timesteps(steps, device=self.device, dtype=noise.dtype)
        else:
            t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=noise.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        solver = Solver(fn, y0, sigma=sigma, temperature=temperature)
        trajectory = solver.integrate(t)
        sampled = trajectory[-1]
        out = sampled

        return out, trajectory
