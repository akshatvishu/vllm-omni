# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/fm/dit.py

import math

import torch
import torch.nn as nn
from x_transformers.x_transformers import RotaryEmbedding

from .modules import DiTBlock, FinalLayer


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        if x.ndim == 0:
            x = x.reshape(1)
        if x.ndim != 1:
            raise ValueError(f"Expected timestep rank-1 [Batch], got {tuple(x.shape)}")
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time


class CondEmbedder(nn.Module):
    def __init__(self, input_feature_size, hidden_size, dropout_prob):
        super().__init__()
        del dropout_prob
        self.cond_embedder = nn.Linear(input_feature_size, hidden_size)

    def forward(self, llm_cond):
        if llm_cond.ndim != 3:
            raise ValueError(f"Expected conditioning rank-3 [Batch, Time, Dimension], got {tuple(llm_cond.shape)}")
        llm_cond = self.cond_embedder(llm_cond)

        return llm_cond


class DiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        hidden_size=1024,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        llm_cond_dim=896,
        cfg_dropout_prob=0.1,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.c_embedder = CondEmbedder(llm_cond_dim, hidden_size, cfg_dropout_prob)
        self.hidden_size = hidden_size
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def forward(self, x, t, c, latent_history, mask=None):
        if x.ndim != 3:
            raise ValueError(f"Expected x rank-3 [Batch, Time, Dimension], got {tuple(x.shape)}")
        if latent_history.ndim != 3:
            raise ValueError(
                f"Expected latent_history rank-3 [Batch, Time, Dimension], got {tuple(latent_history.shape)}"
            )
        if c.ndim != 3:
            raise ValueError(f"Expected conditioning rank-3 [Batch, Time, Dimension], got {tuple(c.shape)}")
        if x.shape[0] != latent_history.shape[0] or x.shape[0] != c.shape[0]:
            raise ValueError(
                "Batch mismatch across x, conditioning, and latent_history: "
                f"{x.shape[0]}, {c.shape[0]}, {latent_history.shape[0]}"
            )
        if x.shape[-1] != self.in_channels:
            raise ValueError(f"x feature dim mismatch: got {x.shape[-1]}, expected {self.in_channels}")
        if latent_history.shape[-1] != self.in_channels:
            raise ValueError(
                f"latent_history feature dim mismatch: got {latent_history.shape[-1]}, expected {self.in_channels}"
            )
        if t.ndim == 0:
            t = t.reshape(1)
        if t.ndim != 1:
            raise ValueError(f"Expected timestep rank-1 [Batch], got {tuple(t.shape)}")
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"Timestep batch mismatch: got {t.shape[0]}, expected {x.shape[0]}")

        t = self.t_embedder(t).unsqueeze(1)
        x_now = self.x_embedder(x)
        x_history = self.x_embedder(latent_history)
        x = torch.cat([x_history, x_now], dim=1)
        c = self.c_embedder(c)
        y = t + c
        x = torch.cat([y, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])

        if mask is not None:
            if mask.ndim != 2:
                raise ValueError(f"Expected mask rank-2 [Batch, Time], got {tuple(mask.shape)}")
            if mask.shape[0] != x_now.shape[0] or mask.shape[1] != x_now.shape[1]:
                raise ValueError(
                    f"Mask shape mismatch: got {tuple(mask.shape)}, expected {(x_now.shape[0], x_now.shape[1])}"
                )
            mask_pad = mask.clone().detach()[:, :1].expand(-1, x_history.shape[1] + c.shape[1])
            mask = torch.cat([mask_pad, mask], dim=-1)
        for block in self.blocks:
            x = block(x, mask, rope)
        x = self.final_layer(x)
        return x

    def forward_with_cfg(self, x, t, c, cfg_scale, latent_history, patch_size):
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if not cfg_scale == 1:
            x = torch.cat([x, x], dim=0)
            latent_history = torch.cat([latent_history, latent_history], dim=0)
            fake_latent = torch.zeros_like(c)
            c = torch.cat([c, fake_latent], dim=0)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        model_out = self.forward(x, t, c, latent_history)
        return model_out[:, -patch_size:, :]
