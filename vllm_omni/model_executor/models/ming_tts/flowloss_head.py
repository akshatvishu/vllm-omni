# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/fm/flowloss.py

import torch
import torch.nn as nn

from .fm.cfm import CFM
from .fm.dit import DiT


class FlowLoss(nn.Module):
    """Diffusion Loss"""

    def __init__(self, z_channels, llm_cond_dim, **kwargs):
        super().__init__()
        self.z_channels = z_channels
        self.cfm = CFM(model=DiT(in_channels=z_channels, llm_cond_dim=llm_cond_dim, **kwargs))

    def sample(self, z, latent_history, cfg=2.0, patch_size=1, sigma=0.25, temperature=0):
        # z: [Batch, Time=1, Dimension]
        # latent_history: [Batch, History_Time, Dimension]
        noise = torch.randn(z.shape[0], self.z_channels, patch_size, device=z.device)
        noise = noise.to(dtype=z.dtype)
        out, _ = self.cfm.sample(
            noise=noise,
            c=z,
            latent_history=latent_history,
            cfg_scale=cfg,
            patch_size=patch_size,
            sigma=sigma,
            temperature=temperature,
        )
        return out
