# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Stable Audio DiT Model for vLLM-Omni.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


def apply_rotary_emb_stable_audio(hidden_states, freqs_cis):
    """
    Applies Rotary Positional Embeddings (RoPE) to the hidden states.

    Unlike standard RoPE which rotates the entire head dimension,
    Stable Audio applies rotation only to the first `rotary_dim` elements of
    the head, concatenating them with the remaining unchanged elements (`x_pass`).
    Computation is forced to float32 to prevent precision degradation in trigonometric ops.
    """
    cos, sin = freqs_cis
    rotary_dim = cos.shape[-1]

    x_rot = hidden_states[..., :rotary_dim]
    x_pass = hidden_states[..., rotary_dim:]

    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    x_real, x_imag = x_rot.reshape(*x_rot.shape[:-1], 2, rotary_dim // 2).unbind(-2)

    x_rotated = torch.cat([-x_imag, x_real], dim=-1)
    x_rot = (x_rot.float() * cos + x_rotated.float() * sin).to(hidden_states.dtype)

    return torch.cat([x_rot, x_pass], dim=-1)


class StableAudioGaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for continuous noise levels (timesteps).

    Projects scalar diffusion timesteps into higher-dimensional periodic
    representations using randomly initialized Fourier features.
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        # Standard initialization. Checkpoint load_weights will overwrite this.
        self.weight = nn.Parameter(
            torch.randn(embedding_size) * scale,
            requires_grad=False,
        )

    def forward(self, x):
        # Run Fourier projection in float32 safely to prevent precision overflow
        x_fp32 = x.float()
        fourier_weight = self.weight.float()
        x_proj = 2 * math.pi * x_fp32[:, None] @ fourier_weight[None, :]

        out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        # Cast back to the model's original dtype
        return out.to(x.dtype)


class StableAudioSelfAttention(nn.Module):
    """
    Optimized self-attention for Stable Audio using vLLM layers.

    Self-attention intentionally uses Multi-Head Attention (MHA),
    meaning all heads are used for Q, K, and V. Grouped-Query Attention (GQA)
    is strictly reserved for cross-attention. This matches the mathematical
    architecture of the original Stable Audio Open model.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        num_key_value_attention_heads,
        attention_head_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=attention_head_dim,
            total_num_heads=num_attention_heads,
            total_num_kv_heads=num_key_value_attention_heads,
            bias=False,
            return_bias=False,
        )
        self.local_heads = self.to_qkv.num_heads
        self.local_kv_heads = self.to_qkv.num_kv_heads

        self.attn = Attention(
            num_heads=self.local_heads,
            head_size=attention_head_dim,
            softmax_scale=1.0 / (attention_head_dim**0.5),
            causal=False,
            num_kv_heads=self.local_kv_heads,
        )
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(self.inner_dim, dim, bias=False, input_is_parallel=True),
                nn.Dropout(dropout),
            ]
        )

    def forward(self, hidden_states, rotary_emb=None):
        B, S, _ = hidden_states.shape

        # Projections: With attn1 set to MHA (24 heads each for Q, K, V),
        # all split sizes are now equal to q_size.
        qkv = self.to_qkv(hidden_states)

        q_size = self.local_heads * self.head_dim

        # Slicing must use equal sizes for Q, K, and V in MHA.
        q, k, v = qkv.split([q_size, q_size, q_size], dim=-1)

        q = q.view(B, S, self.local_heads, self.head_dim)
        k = k.view(B, S, self.local_heads, self.head_dim)
        v = v.view(B, S, self.local_heads, self.head_dim)

        if rotary_emb is not None:
            # Rotary embeddings are applied only to the first rotary_dim.
            q = apply_rotary_emb_stable_audio(q, rotary_emb)
            k = apply_rotary_emb_stable_audio(k, rotary_emb)

        # Kernel Execution: SDPA backend handles the sharded tensors.
        hidden_states = self.attn(q, k, v)
        hidden_states = hidden_states.view(B, S, -1)

        # Output Projection: RowParallelLinear all-reduces to full dim.
        hidden_states, _ = self.to_out[0](hidden_states)

        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class StableAudioCrossAttention(nn.Module):
    """
    Optimized cross-attention for Stable Audio using vLLM layers.

    Cross-attention utilizes Grouped-Query Attention (GQA).
    To support Tensor Parallelism with the SDPA backend, the
    sharded (local) K/V heads are manually expanded to match the local Q heads
    before computing attention.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        num_key_value_attention_heads,
        attention_head_dim,
        cross_attention_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        kv_size = num_key_value_attention_heads * attention_head_dim

        self.to_q = ColumnParallelLinear(dim, num_attention_heads * attention_head_dim, bias=False, gather_output=False)
        self.to_kv = MergedColumnParallelLinear(
            cross_attention_dim,
            [kv_size, kv_size],
            bias=False,
            gather_output=False,
        )

        self.local_heads = self.to_q.output_size_per_partition // attention_head_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.local_kv_heads = num_key_value_attention_heads // tp_size

        self.attn = Attention(
            num_heads=self.local_heads,
            head_size=attention_head_dim,
            num_kv_heads=self.local_heads,
            softmax_scale=1.0 / (attention_head_dim**0.5),
            causal=False,
        )

        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(self.inner_dim, dim, bias=False, input_is_parallel=True),
                nn.Dropout(dropout),
            ]
        )

    def forward(self, hidden_states, encoder_hidden_states):
        B, Sq, _ = hidden_states.shape
        Sk = encoder_hidden_states.shape[1]

        q, _ = self.to_q(hidden_states)

        kv, _ = self.to_kv(encoder_hidden_states)

        kv_size = self.local_kv_heads * self.head_dim
        k, v = kv.split([kv_size, kv_size], dim=-1)

        q = q.view(B, Sq, self.local_heads, self.head_dim)
        k = k.view(B, Sk, self.local_kv_heads, self.head_dim)
        v = v.view(B, Sk, self.local_kv_heads, self.head_dim)

        # GQA expansion
        # Expand LOCAL KV to match LOCAL Q heads.
        if self.local_kv_heads != self.local_heads:
            num_groups = self.local_heads // self.local_kv_heads

            k = k.unsqueeze(3).expand(-1, -1, -1, num_groups, -1)
            k = k.reshape(B, Sk, self.local_heads, self.head_dim)

            v = v.unsqueeze(3).expand(-1, -1, -1, num_groups, -1)
            v = v.reshape(B, Sk, self.local_heads, self.head_dim)

        hidden_states = self.attn(q, k, v)

        hidden_states = hidden_states.view(B, Sq, -1)

        # Output projection (RowParallel → all-reduce)
        hidden_states, _ = self.to_out[0](hidden_states)

        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class StableAudioFeedForward(nn.Module):
    """
    Tensor-parallel implementation of the SwiGLU feed-forward network.

    To reduce GPU communication overhead, the original separate projection
    and gate linear layers are fused into a single `MergedColumnParallelLinear`
    layer. The combined output is chunked into (hidden, gate), the gate is
    activated with SiLU, and the elementwise product implements SwiGLU.

    The final projection uses `RowParallelLinear`, which performs the
    cross-rank reduction required in tensor parallelism.
    """

    def __init__(self, dim, inner_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.ModuleDict(
                {
                    "proj": MergedColumnParallelLinear(
                        dim,
                        [inner_dim, inner_dim],  # [hidden_size, gate_size]
                        bias=True,
                        gather_output=False,
                    )
                }
            ),
            RowParallelLinear(
                inner_dim,
                dim,
                bias=True,
                input_is_parallel=True,
            ),
        )

    def forward(self, hidden_states):
        # ColumnParallelLinear
        hidden_states, _ = self.net[0]["proj"](hidden_states)

        hidden, gate = hidden_states.chunk(2, dim=-1)

        hidden_states = hidden * torch.nn.functional.silu(gate)

        # RowParallelLinear
        hidden_states, _ = self.net[1](hidden_states)

        return hidden_states


class StableAudioDiTBlock(nn.Module):
    """
    A single Diffusion Transformer (DiT) block for Stable Audio Open.

    Applies three components sequentially with residual connections:
    1. Self-attention (multi-head, with rotary embeddings)
    2. Cross-attention (grouped-query, attends to text/duration conditioning)
    3. Feed-forward (SwiGLU, expands dimension by `ff_mult`)
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        num_key_value_attention_heads,
        attention_head_dim,
        cross_attention_dim,
        ff_mult=4,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        # Self-attention in Stable Audio Open uses MHA (num_heads == num_kv_heads)
        self.attn1 = StableAudioSelfAttention(
            dim,
            num_attention_heads,
            num_attention_heads,
            attention_head_dim,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = StableAudioCrossAttention(
            dim,
            num_attention_heads,
            num_key_value_attention_heads,
            attention_head_dim,
            cross_attention_dim,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.ff = StableAudioFeedForward(dim, dim * ff_mult)

    def forward(self, hidden_states, encoder_hidden_states, rotary_embedding=None):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn1(hidden_states, rotary_embedding)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(hidden_states, encoder_hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class StableAudioDiTModel(nn.Module):
    """
    Diffusion Transformer (DiT) model for Stable Audio Open.

    Processes audio latents of shape [B, C, T], prepends global and
    timestep embeddings, applies a stack of DiT blocks, and projects
    back to the original channel dimension.

    Linear projections are implemented using tensor-parallel vLLM
    layers to support multi-GPU execution.
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        sample_size: int = 1024,
        in_channels: int = 64,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        num_key_value_attention_heads: int = 12,
        out_channels: int = 64,
        cross_attention_dim: int = 768,
        time_proj_dim: int = 256,
        global_states_input_dim: int = 1536,
        cross_attention_input_dim: int = 768,
    ):
        super().__init__()

        tp_size = get_tensor_model_parallel_world_size()

        if num_attention_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by tensor parallel size ({tp_size})."
            )
        if num_key_value_attention_heads % tp_size != 0:
            raise ValueError(
                f"num_key_value_attention_heads ({num_key_value_attention_heads}) must be "
                f"divisible by tensor parallel size ({tp_size})."
            )

        self.inner_dim = num_attention_heads * attention_head_dim

        self.config = type(
            "Config",
            (),
            {
                "sample_size": sample_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "num_layers": num_layers,
                "attention_head_dim": attention_head_dim,
                "num_attention_heads": num_attention_heads,
                "num_key_value_attention_heads": num_key_value_attention_heads,
                "cross_attention_dim": cross_attention_dim,
                "time_proj_dim": time_proj_dim,
                "global_states_input_dim": global_states_input_dim,
                "cross_attention_input_dim": cross_attention_input_dim,
            },
        )()

        self.activation = nn.SiLU()

        # Time embedding
        self.time_proj = StableAudioGaussianFourierProjection(embedding_size=time_proj_dim // 2)
        self.timestep_proj_0 = ColumnParallelLinear(
            time_proj_dim,
            self.inner_dim,
            bias=True,
            gather_output=False,
        )
        self.timestep_proj_2 = RowParallelLinear(
            self.inner_dim,
            self.inner_dim,
            bias=True,
            input_is_parallel=True,
            reduce_results=True,
        )

        # Global embedding
        self.global_proj_0 = ColumnParallelLinear(
            global_states_input_dim,
            self.inner_dim,
            bias=False,
            gather_output=False,
        )
        self.global_proj_2 = RowParallelLinear(
            self.inner_dim,
            self.inner_dim,
            bias=False,
            input_is_parallel=True,
            # Force All-Reduce to full-dimension. Must match timestep_proj_2
            # so they can be added and concatenated with the un-sharded global token.
            reduce_results=True,
        )

        self.cross_attention_proj = nn.Sequential(
            nn.Linear(cross_attention_input_dim, cross_attention_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cross_attention_dim, cross_attention_dim, bias=False),
        )

        self.preprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.proj_in = ColumnParallelLinear(
            in_channels,
            self.inner_dim,
            bias=False,
            # Force full-dimension output so it can be concatenated with the full global token
            gather_output=True,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                StableAudioDiTBlock(
                    self.inner_dim,
                    num_attention_heads,
                    num_key_value_attention_heads,
                    attention_head_dim,
                    cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Each block's to_out RowParallelLinear all-reduces back to FULL dim,
        # so proj_out receives a FULL tensor → input_is_parallel=False.
        self.proj_out = ReplicatedLinear(
            self.inner_dim,
            out_channels,
            bias=False,
        )

        self.postprocess_conv = nn.Conv1d(out_channels, out_channels, 1, bias=False)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        global_hidden_states,
        rotary_embedding=None,
        return_dict=True,
    ):
        # Cross attention conditioning (replicated)
        cross_attention_hidden_states = self.cross_attention_proj(encoder_hidden_states)

        # Time embedding
        time_hidden_states = self.time_proj(timestep)

        # MLP
        time_hidden_states, _ = self.timestep_proj_0(time_hidden_states)
        time_hidden_states = self.activation(time_hidden_states)
        time_hidden_states, _ = self.timestep_proj_2(time_hidden_states)

        # Global embedding  [B, 1, global_dim] → sharded [B, 1, shard_dim]
        global_hidden_states, _ = self.global_proj_0(global_hidden_states)

        global_hidden_states = self.activation(global_hidden_states)
        global_hidden_states, _ = self.global_proj_2(global_hidden_states)

        # Invariant: timestep and global are FULL (reduce_results=True), proj_in is FULL (gather_output=True)
        if time_hidden_states.shape[-1] != self.inner_dim:
            raise RuntimeError(
                f"time_hidden_states dimension mismatch: expected {self.inner_dim}, got {time_hidden_states.shape[-1]}"
            )
        if global_hidden_states.shape[-1] != self.inner_dim:
            raise RuntimeError(
                f"global_hidden_states dimension mismatch: expected {self.inner_dim}, "
                f"got {global_hidden_states.shape[-1]}"
            )

        global_hidden_states = global_hidden_states + time_hidden_states.unsqueeze(1)

        # Audio latent  [B, C, T] → sharded [B, T, shard_dim]
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states, _ = self.proj_in(hidden_states)

        # Prepend global token: [B, 1+T, inner_dim]
        hidden_states = torch.cat([global_hidden_states, hidden_states], dim=1)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                cross_attention_hidden_states,
                rotary_embedding,
            )

        # Output projection  FULL [B, 1+T, inner_dim] → [B, 1+T, out_channels]
        hidden_states, _ = self.proj_out(hidden_states)

        # Drop global token, restore [B, C, T]
        hidden_states = hidden_states.transpose(1, 2)[:, :, 1:]
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        if return_dict:
            return Transformer2DModelOutput(sample=hidden_states)
        return (hidden_states,)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        tp_rank = get_tensor_model_parallel_rank()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        name_mapping = {
            "timestep_proj.0.weight": "timestep_proj_0.weight",
            "timestep_proj.0.bias": "timestep_proj_0.bias",
            "timestep_proj.2.weight": "timestep_proj_2.weight",
            "timestep_proj.2.bias": "timestep_proj_2.bias",
            "global_proj.0.weight": "global_proj_0.weight",
            "global_proj.2.weight": "global_proj_2.weight",
            # legacy keys
            "timestep_proj.linear_1.weight": "timestep_proj_0.weight",
            "timestep_proj.linear_1.bias": "timestep_proj_0.bias",
            "timestep_proj.linear_2.weight": "timestep_proj_2.weight",
            "timestep_proj.linear_2.bias": "timestep_proj_2.bias",
            "global_proj.linear_1.weight": "global_proj_0.weight",
            "global_proj.linear_1.bias": "global_proj_0.bias",
            "global_proj.linear_2.weight": "global_proj_2.weight",
            "global_proj.linear_2.bias": "global_proj_2.bias",
        }

        def remap_name(n: str) -> str:
            n = name_mapping.get(n, n)
            if ".ff.net.2." in n:
                n = n.replace(".ff.net.2.", ".ff.net.1.")
            return n

        qkv_buffer: dict[str, dict[str, torch.Tensor]] = {}

        for name, loaded_weight in weights:
            mapped_name = remap_name(name)

            # Self-attention QKV fusion

            if ".attn1.to_" in mapped_name and any(x in mapped_name for x in ("q.", "k.", "v.")):
                shard_id = "q" if ".to_q." in mapped_name else ("k" if ".to_k." in mapped_name else "v")
                fused_name = mapped_name.replace(f".to_{shard_id}.", ".to_qkv.")
                qkv_buffer.setdefault(fused_name, {})[shard_id] = loaded_weight

                if len(qkv_buffer[fused_name]) == 3:
                    if fused_name not in params_dict:
                        logger.error(f"[QKV ERROR] Missing fused param {fused_name}")
                        continue
                    param = params_dict[fused_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)

                    q_shape = qkv_buffer[fused_name]["q"].shape
                    k_shape = qkv_buffer[fused_name]["k"].shape
                    v_shape = qkv_buffer[fused_name]["v"].shape
                    if not (q_shape == k_shape == v_shape):
                        logger.error(
                            f"[QKV SHAPE MISMATCH][rank={tp_rank}] {fused_name} Q={q_shape} K={k_shape} V={v_shape}"
                        )
                    for sid in ("q", "k", "v"):
                        weight_loader(param, qkv_buffer[fused_name][sid], sid)
                    loaded_params.add(fused_name)
                    del qkv_buffer[fused_name]
                continue

            #  Cross-attention Q

            if ".attn2.to_q." in mapped_name:
                if mapped_name not in params_dict:
                    logger.error(f"[ATTN2-Q] Missing param {mapped_name}")
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped_name)
                continue

            #  Cross-attention KV fusion

            if any(f".attn2.{x}.weight" in mapped_name for x in ("to_k", "to_v")):
                shard_id = 0 if ".to_k." in mapped_name else 1
                fused_name = mapped_name.replace(".to_k.weight", ".to_kv.weight").replace(
                    ".to_v.weight", ".to_kv.weight"
                )

                if fused_name not in params_dict:
                    continue
                param = params_dict[fused_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                weight_loader(param, loaded_weight, loaded_shard_id=shard_id)
                loaded_params.add(fused_name)
                continue

            #  GLU FFN

            if ".ff.net.0.proj." in mapped_name:
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                hidden_w, gate_w = loaded_weight.chunk(2, dim=0)

                weight_loader(param, hidden_w, loaded_shard_id=0)
                weight_loader(param, gate_w, loaded_shard_id=1)

                loaded_params.add(mapped_name)
                continue

            # Standard loader

            if mapped_name in params_dict:
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped_name)

        return loaded_params
