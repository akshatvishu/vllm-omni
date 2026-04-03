# Based on https://github.com/inclusionAI/Ming-omni-tts/blob/main/fm/flowloss.py

import torch
import torch.nn as nn

from .CFM import CFM
from .dit import DiT


class FlowLoss(nn.Module):
    """Diffusion Loss"""

    def __init__(self, z_channels, llm_cond_dim, **kwargs):
        super().__init__()
        self.z_channels = z_channels
        self.cfm = CFM(model=DiT(in_channels=z_channels, llm_cond_dim=llm_cond_dim, **kwargs))

    def forward(self, cond, target, latent_history, mask, patch_size):
        return self.cfm(cond=cond, target=target, latent_history=latent_history, mask=mask, patch_size=patch_size)

    def sample(self, z, latent_history, cfg=2.0, patch_size=1, sigma=0.25, temperature=0):
        if z.ndim != 3:
            raise ValueError(f"Expected z rank-3 [Batch, Time, Dimension], got {tuple(z.shape)}")
        if z.shape[1] != 1:
            raise ValueError(f"Expected z time dim to be 1 for Ming dense decode, got {z.shape[1]}")
        if latent_history.ndim != 3:
            raise ValueError(
                f"Expected latent_history rank-3 [Batch, Time, Dimension], got {tuple(latent_history.shape)}"
            )
        if z.shape[0] != latent_history.shape[0]:
            raise ValueError(f"Batch mismatch: z batch={z.shape[0]} vs latent_history batch={latent_history.shape[0]}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if not torch.isfinite(z).all():
            raise RuntimeError("Non-finite conditioning z in FlowLoss.sample().")
        if not torch.isfinite(latent_history).all():
            raise RuntimeError("Non-finite latent_history in FlowLoss.sample().")
        noise = torch.randn(z.shape[0], self.z_channels, patch_size, device=z.device)
        if not torch.isfinite(noise).all():
            raise RuntimeError("Non-finite noise in FlowLoss.sample().")
        noise = noise.to(dtype=z.dtype)  # match conditioning dtype — no autocast in vllm-omni
        out, _ = self.cfm.sample(
            noise=noise,
            c=z,
            latent_history=latent_history,
            cfg_scale=cfg,
            patch_size=patch_size,
            sigma=sigma,
            temperature=temperature,
        )
        # out shape: [B, patch_size, z_channels]
        return out
