# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# mypy: disable-error-code=attr-defined
"""Forward references copied from vLLM-Omni 43e507117f04a86f2df8d405cbe03c1b1642eac9."""

from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_npu_available
from torch.nn.utils.rnn import pad_sequence
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope_min_tokens
from vllm_omni.diffusion.models.flux.flux_transformer import logger

_FUSED_MIN_TOKENS = 512
_FUSED_QK_ROPE = HAS_TRITON and current_platform.is_cuda()
SEQ_MULTI_OF = 32
LEARNED_PADDING = "learned"
ZERO_MASKED_PADDING = "zero_masked"


def _fusion_enabled(sequence_parallel_size: int | None, *, enforce_eager: bool) -> bool:
    return (
        _FUSED_QK_ROPE
        and enforce_eager
        and not torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
        and not (sequence_parallel_size is not None and sequence_parallel_size > 1)
    )


def _prepare_rotary_emb(
    txt_cos: torch.Tensor,
    txt_sin: torch.Tensor,
    img_cos: torch.Tensor,
    img_sin: torch.Tensor,
    *,
    enable_fusion: bool,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, int]:
    joint = torch.cat((txt_cos, img_cos), dim=0), torch.cat((txt_sin, img_sin), dim=0)
    if not enable_fusion:
        return joint
    return joint[0], joint[1], fused_qk_norm_rope_min_tokens(_FUSED_MIN_TOKENS)


def pinned_flux_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
) -> torch.Tensor | Transformer2DModelOutput:
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        img_ids: (`torch.Tensor`):
            The position ids for image tokens.
        txt_ids (`torch.Tensor`):
            The position ids for text tokens.
        guidance (`torch.Tensor`):
            Guidance embeddings for guidance-distilled variant of the model.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """

    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

    if guidance is not None:
        guidance = guidance.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    if is_torch_npu_available():
        freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
        image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
    else:
        image_rotary_emb = self.pos_embed(ids)

    for index_block, block in enumerate(self.transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    for index_block, block in enumerate(self.single_transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def pinned_longcat_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    return_dict: bool = True,
) -> torch.FloatTensor | Transformer2DModelOutput:
    fwd_context = get_forward_context()
    sp_size = self.parallel_config.sequence_parallel_size
    if sp_size is not None and sp_size > 1:
        fwd_context.split_text_embed_in_sp = False

    # Hidden states are sharded prior to forward() when sp is active
    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000

    temb = self.time_embed(timestep, hidden_states.dtype)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # Compute RoPE embeddings via rope_preparer module
    # _sp_plan will automatically shard img_cos/img_sin (outputs 2, 3)
    # txt_cos/txt_sin (outputs 0, 1) remain replicated for dual-stream attention
    txt_cos, txt_sin, img_cos, img_sin = self.rope_preparer(txt_ids, img_ids)

    # Preserve the ordinary full-width tables.  A third scalar only marks
    # the eligible eager SP=1 route and resolves the shared threshold once
    # for all attention blocks in this transformer call.
    image_rotary_emb = _prepare_rotary_emb(
        txt_cos,
        txt_sin,
        img_cos,
        img_sin,
        enable_fusion=_fusion_enabled(sp_size, enforce_eager=self.enforce_eager),
    )

    for block in self.transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )

    for block in self.single_transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )

    hidden_states = self.norm_out(hidden_states, temb)

    # proj_out gathers for sequence parallel
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def pinned_stable_audio_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    global_hidden_states: torch.Tensor | None = None,
    rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
    return_dict: bool = True,
    attention_mask: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | Transformer2DModelOutput:
    """
    Forward pass of the Stable Audio DiT model.

    Args:
        hidden_states: Input latent tensor [B, C, L] (C=in_channels=64)
        timestep: Timestep tensor [B] or [1]
        encoder_hidden_states: Text/condition embeddings [B, S, D]
        global_hidden_states: Global conditioning (duration) [B, 1, D]
        rotary_embedding: Precomputed rotary embeddings (cos, sin)
        return_dict: Whether to return a dataclass or tuple
        attention_mask: Attention mask for self-attention
        encoder_attention_mask: Attention mask for cross-attention

    Returns:
        Denoised latent tensor
    """
    # Project cross-attention inputs
    cross_attention_hidden_states = self.cross_attention_proj(encoder_hidden_states)

    # Global embedding projection [B, 1, D] -> [B, 1, inner_dim]
    global_hidden_states = self.global_proj(global_hidden_states)

    # Time embedding: timestep -> time_proj -> timestep_proj
    time_hidden_states = self.timestep_proj(self.time_proj(timestep.to(self.dtype)))

    # Combine global and time embeddings [B, 1, inner_dim]
    global_hidden_states = global_hidden_states + time_hidden_states.unsqueeze(1)

    # Pre-process with residual: [B, C, L]
    hidden_states = self.preprocess_conv(hidden_states) + hidden_states

    # Transpose: [B, C, L] -> [B, L, C]
    hidden_states = hidden_states.transpose(1, 2)

    # Project to inner_dim: [B, L, C] -> [B, L, inner_dim]
    hidden_states = self.proj_in(hidden_states)

    # Prepend global states to hidden states: [B, 1+L, inner_dim]
    hidden_states = torch.cat([global_hidden_states, hidden_states], dim=1)

    # Update attention mask if provided
    if attention_mask is not None:
        prepend_mask = torch.ones(
            (hidden_states.shape[0], 1),
            device=hidden_states.device,
            dtype=torch.bool,
        )
        attention_mask = torch.cat([prepend_mask, attention_mask], dim=-1)

    # Transformer blocks
    for block in self.transformer_blocks:
        hidden_states = block(
            hidden_states,
            cross_attention_hidden_states,
            rotary_embedding=rotary_embedding,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

    # Project back to out_channels: [B, 1+L, inner_dim] -> [B, 1+L, out_channels]
    hidden_states = self.proj_out(hidden_states)

    # Transpose and remove prepended global token: [B, L, C] -> [B, C, L]
    hidden_states = hidden_states.transpose(1, 2)[:, :, 1:]

    # Post-process with residual: [B, C, L]
    hidden_states = self.postprocess_conv(hidden_states) + hidden_states

    if return_dict:
        return Transformer2DModelOutput(sample=hidden_states)
    return (hidden_states,)


def pinned_flux2_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
) -> torch.Tensor | Transformer2DModelOutput:
    joint_attention_kwargs = joint_attention_kwargs or {}

    num_txt_tokens = encoder_hidden_states.shape[1]
    sp_size = self.parallel_config.sequence_parallel_size
    if sp_size and sp_size > 1 and is_forward_context_available():
        get_forward_context().split_text_embed_in_sp = False

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = self.time_guidance_embed(timestep, guidance)

    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)[0]

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    if is_torch_npu_available():
        txt_freqs_cos, txt_freqs_sin, img_freqs_cos, img_freqs_sin = self.rope_prepare(img_ids.cpu(), txt_ids.cpu())
        txt_freqs_cos = txt_freqs_cos.npu()
        txt_freqs_sin = txt_freqs_sin.npu()
        img_freqs_cos = img_freqs_cos.npu()
        img_freqs_sin = img_freqs_sin.npu()
    else:
        txt_freqs_cos, txt_freqs_sin, img_freqs_cos, img_freqs_sin = self.rope_prepare(img_ids, txt_ids)
    concat_rotary_emb = (
        torch.cat([txt_freqs_cos, img_freqs_cos], dim=0),
        torch.cat([txt_freqs_sin, img_freqs_sin], dim=0),
    )

    # Create separate masks for image and text portions for Ulysses SP joint attention
    hidden_states_mask = None
    encoder_hidden_states_mask = None
    if is_forward_context_available():
        ctx = get_forward_context()
    else:
        ctx = None
    if (
        ctx is not None
        and self.parallel_config.sequence_parallel_size > 1
        and self.parallel_config.mask_sp_padding
        and ctx.sp_original_seq_len is not None
        and ctx.sp_padding_size > 0
    ):
        batch_size = hidden_states.shape[0]
        img_padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
        hidden_states_mask = torch.ones(
            batch_size,
            img_padded_seq_len,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        hidden_states_mask[:, ctx.sp_original_seq_len :] = False
        if hidden_states_mask.all():
            hidden_states_mask = None
    elif (
        ctx is not None
        and self.parallel_config.sequence_parallel_size > 1
        and not self.parallel_config.mask_sp_padding
        and ctx.sp_original_seq_len is not None
        and ctx.sp_padding_size > 0
    ):
        logger.warning_once(
            "SP auto-padding applied %d token(s) (seq_len=%d, ulysses_degree=%d). "
            "Padding tokens are not masked from attention (mask_sp_padding=False), "
            "which avoids the varlen attention path but may produce minor numerical differences. "
            "Set parallel_config.mask_sp_padding=True to restore strict masking.",
            ctx.sp_padding_size,
            ctx.sp_original_seq_len,
            self.parallel_config.sequence_parallel_size,
        )

    if hidden_states_mask is not None:
        joint_attention_kwargs["hidden_states_mask"] = hidden_states_mask
    if encoder_hidden_states_mask is not None:
        joint_attention_kwargs["encoder_hidden_states_mask"] = encoder_hidden_states_mask

    for index_block, block in enumerate(self.transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=double_stream_mod_img,
            temb_mod_params_txt=double_stream_mod_txt,
            image_rotary_emb=concat_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb_mod_params=single_stream_mod,
            image_rotary_emb=concat_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            text_seq_len=num_txt_tokens,
        )

    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def pinned_flux2_klein_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
) -> torch.Tensor | Transformer2DModelOutput:
    joint_attention_kwargs = joint_attention_kwargs or {}

    num_txt_tokens = encoder_hidden_states.shape[1]

    sp_size = self.parallel_config.sequence_parallel_size
    if sp_size is not None and sp_size > 1:
        get_forward_context().split_text_embed_in_sp = False

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = self.time_guidance_embed(timestep, guidance)

    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)[0]

    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    txt_freqs_cos, txt_freqs_sin, img_freqs_cos, img_freqs_sin = self.rope_prepare(img_ids, txt_ids)

    concat_rotary_emb = (
        torch.cat([txt_freqs_cos, img_freqs_cos], dim=0),
        torch.cat([txt_freqs_sin, img_freqs_sin], dim=0),
    )

    # Create separate masks for image and text portions for Ulysses SP joint attention
    hidden_states_mask = None
    encoder_hidden_states_mask = None
    ctx = get_forward_context()
    if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
        batch_size = hidden_states.shape[0]
        img_padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size

        hidden_states_mask = torch.ones(
            batch_size,
            img_padded_seq_len,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        hidden_states_mask[:, ctx.sp_original_seq_len :] = False
        if hidden_states_mask.all():
            hidden_states_mask = None

    if hidden_states_mask is not None:
        joint_attention_kwargs["hidden_states_mask"] = hidden_states_mask
    if encoder_hidden_states_mask is not None:
        joint_attention_kwargs["encoder_hidden_states_mask"] = encoder_hidden_states_mask

    for block in self.transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=double_stream_mod_img,
            temb_mod_params_txt=double_stream_mod_txt,
            image_rotary_emb=concat_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for block in self.single_transformer_blocks:
        hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb_mod_params=single_stream_mod,
            image_rotary_emb=concat_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            text_seq_len=num_txt_tokens,
        )

    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def pinned_qwen_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: list[tuple[int, int, int]] | None = None,
    txt_seq_lens: list[int] | None = None,
    guidance: torch.Tensor = None,  # TODO: this should probably be removed
    attention_kwargs: dict[str, Any] | None = None,
    additional_t_cond=None,
    return_dict: bool = True,
) -> torch.Tensor | Transformer2DModelOutput:
    """
    The [`QwenTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
            Mask of the input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    # if attention_kwargs is not None:
    #     attention_kwargs = attention_kwargs.copy()
    #     lora_scale = attention_kwargs.pop("scale", 1.0)
    # else:
    #     lora_scale = 1.0

    # Set split_text_embed_in_sp = False for dual-stream attention
    # QwenImage uses *dual-stream* (text + image) and runs a *joint attention*.
    # Text embeddings must be replicated across SP ranks for correctness.
    if self.parallel_config.sequence_parallel_size > 1:
        get_forward_context().split_text_embed_in_sp = False

    # Prepare hidden_states and RoPE via ImageRopePrepare module
    # _sp_plan will shard hidden_states and vid_freqs together via split_output=True
    # txt_freqs is kept replicated for dual-stream attention
    hidden_states, vid_freqs, txt_freqs = self.image_rope_prepare(hidden_states, img_shapes, txt_seq_lens)
    image_rotary_emb = (vid_freqs, txt_freqs)

    # Ensure timestep tensor is on the same device and dtype as hidden_states
    timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)

    # Prepare timestep and modulate_index via ModulateIndexPrepare module
    # _sp_plan will shard modulate_index via split_output=True (when zero_cond_t=True)
    # This ensures modulate_index sequence dimension matches sharded hidden_states
    timestep, modulate_index = self.modulate_index_prepare(timestep, img_shapes)

    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )

    # Check for SP auto_pad: create attention mask dynamically if padding was applied
    # In Ulysses mode, attention is computed on the FULL sequence (after All-to-All)
    hidden_states_mask = None  # default
    ctx = get_forward_context()
    if (
        self.parallel_config is not None
        and self.parallel_config.sequence_parallel_size > 1
        and self.parallel_config.mask_sp_padding
        and ctx.sp_original_seq_len is not None
        and ctx.sp_padding_size > 0
    ):
        # Create mask for the full (padded) sequence
        # valid positions = True, padding positions = False
        batch_size = hidden_states.shape[0]
        padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
        hidden_states_mask = torch.ones(
            batch_size,
            padded_seq_len,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        hidden_states_mask[:, ctx.sp_original_seq_len :] = False
        if hidden_states_mask.all():
            hidden_states_mask = None
    elif (
        self.parallel_config is not None
        and self.parallel_config.sequence_parallel_size > 1
        and not self.parallel_config.mask_sp_padding
        and ctx.sp_original_seq_len is not None
        and ctx.sp_padding_size > 0
    ):
        logger.warning_once(
            "SP auto-padding applied %d token(s) (seq_len=%d, ulysses_degree=%d). "
            "Padding tokens are not masked from attention (mask_sp_padding=False), "
            "which avoids the varlen attention path but may produce minor numerical differences. "
            "Set parallel_config.mask_sp_padding=True to restore strict masking.",
            ctx.sp_padding_size,
            ctx.sp_original_seq_len,
            self.parallel_config.sequence_parallel_size,
        )

    if encoder_hidden_states_mask is not None and encoder_hidden_states_mask.all():
        encoder_hidden_states_mask = None

    for index_block, block in enumerate(self.transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=attention_kwargs,
            modulate_index=modulate_index,
            hidden_states_mask=hidden_states_mask,
        )

    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]
    # Use only the image part (hidden_states) from the dual-stream blocks
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    # Note: SP gather is handled automatically by _sp_plan's SequenceParallelGatherHook
    # on proj_out output. No manual all_gather needed here.

    return Transformer2DModelOutput(sample=output)


def pinned_z_image_forward(
    self,
    x: list[torch.Tensor],
    t,
    cap_feats: list[torch.Tensor],
    patch_size=2,
    f_patch_size=1,
    ref_x: list[torch.Tensor | None] | None = None,
    cap_feats_2: list[torch.Tensor] | None = None,
):
    assert patch_size in self.all_patch_size
    assert f_patch_size in self.all_f_patch_size

    bsz = len(x)
    device = x[0].device
    t = t * self.t_scale
    t = self.t_embedder(t)

    (
        x,
        cap_feats,
        x_size,
        x_pos_ids,
        cap_pos_ids,
        x_inner_pad_mask,
        cap_inner_pad_mask,
        cap_feats_2,
    ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size, ref_x, cap_feats_2)

    # x embed & refine
    x_item_seqlens = [len(_) for _ in x]
    assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
    x_max_item_seqlen = max(x_item_seqlens)

    x = torch.cat(x, dim=0)
    x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

    # Match t_embedder output dtype to x for layerwise casting compatibility
    adaln_input = t.type_as(x)
    # Use torch.where instead of x[mask]= to avoid aten::index_put_/nonzero and cudaStreamSynchronize
    x_pad_mask = torch.cat(x_inner_pad_mask)
    x_padding = (
        self.x_pad_token.expand(x.shape[0], -1)
        if self.alignment_padding_mode == LEARNED_PADDING
        else torch.zeros_like(x)
    )
    x = torch.where(
        x_pad_mask.unsqueeze(1).expand_as(x),
        x_padding,
        x,
    )
    x = list(x.split(x_item_seqlens, dim=0))
    x_cos, x_sin = self.rope_embedder(torch.cat(x_pos_ids, dim=0))
    x_cos = list(x_cos.split(x_item_seqlens, dim=0))
    x_sin = list(x_sin.split(x_item_seqlens, dim=0))

    x = pad_sequence(x, batch_first=True, padding_value=0.0)
    x_cos = pad_sequence(x_cos, batch_first=True, padding_value=0.0)
    x_sin = pad_sequence(x_sin, batch_first=True, padding_value=0.0)
    x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(x_item_seqlens):
        x_attn_mask[i, :seq_len] = 1
        if self.alignment_padding_mode == ZERO_MASKED_PADDING:
            x_attn_mask[i, :seq_len].masked_fill_(x_inner_pad_mask[i], False)

    for layer in self.noise_refiner:
        x = layer(x, x_attn_mask, x_cos, x_sin, adaln_input)

    # cap embed & refine
    cap_item_seqlens = [len(_) for _ in cap_feats]
    cap_feats = torch.cat(cap_feats, dim=0)
    cap_feats = self.cap_embedder(cap_feats)
    if cap_feats_2:
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        if len(cap_feats) != len(cap_feats_2):
            raise ValueError("Primary and direct caption conditions must have equal batch size.")
        cap_feats = [torch.cat([primary, direct], dim=0) for primary, direct in zip(cap_feats, cap_feats_2)]
        cap_item_seqlens = [len(item) for item in cap_feats]
        cap_feats = torch.cat(cap_feats, dim=0)
    assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
    cap_max_item_seqlen = max(cap_item_seqlens)

    # Use torch.where instead of cap_feats[mask]= to avoid aten::index_put_/nonzero and cudaStreamSynchronize
    cap_pad_mask = torch.cat(cap_inner_pad_mask)
    cap_padding = (
        self.cap_pad_token.expand(cap_feats.shape[0], -1)
        if self.alignment_padding_mode == LEARNED_PADDING
        else torch.zeros_like(cap_feats)
    )
    cap_feats = torch.where(
        cap_pad_mask.unsqueeze(1).expand_as(cap_feats),
        cap_padding,
        cap_feats,
    )
    cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
    cap_cos, cap_sin = self.rope_embedder(torch.cat(cap_pos_ids, dim=0))
    cap_cos = list(cap_cos.split(cap_item_seqlens, dim=0))
    cap_sin = list(cap_sin.split(cap_item_seqlens, dim=0))

    cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
    cap_cos = pad_sequence(cap_cos, batch_first=True, padding_value=0.0)
    cap_sin = pad_sequence(cap_sin, batch_first=True, padding_value=0.0)
    cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(cap_item_seqlens):
        cap_attn_mask[i, :seq_len] = 1
        if self.alignment_padding_mode == ZERO_MASKED_PADDING:
            cap_attn_mask[i, :seq_len].masked_fill_(cap_inner_pad_mask[i], False)

    for layer in self.context_refiner:
        cap_feats = layer(cap_feats, cap_attn_mask, cap_cos, cap_sin)

    # Prepare unified tensors via UnifiedPrepare module
    # This enables _cp_plan to shard outputs via split_output=True
    unified, unified_cos, unified_sin, unified_attn_mask = self.unified_prepare(
        x,
        x_cos,
        x_sin,
        cap_feats,
        cap_cos,
        cap_sin,
        x_item_seqlens,
        cap_item_seqlens,
        x_attn_mask,
        cap_attn_mask,
    )
    # Main transformer blocks
    for layer in self.layers:
        unified = layer(unified, unified_attn_mask, unified_cos, unified_sin, adaln_input)

    # Final layer
    unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)

    unified = list(unified.unbind(dim=0))
    x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

    return x, {}
