# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Model-specific extractors for TeaCache.

This module provides a registry of extractor functions that know how to extract
modulated inputs from different transformer architectures. Adding support for
a new model requires only adding a new extractor function to the registry.

With Option B enhancement, extractors now return a CacheContext object containing
all model-specific information needed for generic caching, including preprocessing,
transformer execution, and postprocessing logic.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.forward_context import get_forward_context

logger = init_logger(__name__)


@dataclass
class CacheContext:
    """
    Context object containing all model-specific information for caching.

    This allows the TeaCacheHook to remain completely generic - all model-specific
    logic is encapsulated in the extractor that returns this context.

    Attributes:
        modulated_input: Tensor used for cache decision (similarity comparison).
            Must be a torch.Tensor extracted from the first transformer block,
            typically after applying normalization and modulation.

        hidden_states: Current hidden states (will be modified by caching).
            Must be a torch.Tensor representing the main image/latent states
            after preprocessing but before transformer blocks.

        encoder_hidden_states: Optional encoder states (for dual-stream models).
            Set to None for single-stream models (e.g., Flux).
            For dual-stream models (e.g., Qwen), contains text encoder outputs.

        temb: Timestep embedding tensor.
            Must be a torch.Tensor containing the timestep conditioning.

        run_transformer_blocks: Callable that executes model-specific transformer blocks.
            Signature: () -> tuple[torch.Tensor, ...]

            Returns:
                tuple containing:
                - [0]: processed hidden_states (required)
                - [1]: processed encoder_hidden_states (optional, only for dual-stream)

            Example for single-stream:
                def run_blocks():
                    h = hidden_states
                    for block in module.transformer_blocks:
                        h = block(h, temb=temb)
                    return (h,)

            Example for dual-stream:
                def run_blocks():
                    h, e = hidden_states, encoder_hidden_states
                    for block in module.transformer_blocks:
                        e, h = block(h, e, temb=temb)
                    return (h, e)

        postprocess: Callable that does model-specific output postprocessing.
            Signature: (torch.Tensor) -> Union[torch.Tensor, Transformer2DModelOutput, tuple]

            Takes the processed hidden_states and applies final transformations
            (normalization, projection) to produce the model output.

            Example:
                def postprocess(h):
                    h = module.norm_out(h, temb)
                    output = module.proj_out(h)
                    return Transformer2DModelOutput(sample=output)

        extra_states: Optional dict for additional model-specific state.
            Use this for models that need to pass additional context beyond
            the standard fields.
    """

    modulated_input: torch.Tensor
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor | None
    temb: torch.Tensor
    run_transformer_blocks: Callable[[], tuple[torch.Tensor, ...]]
    postprocess: Callable[[torch.Tensor], Any]
    extra_states: dict[str, Any] | None = None

    def validate(self) -> None:
        """
        Validate that the CacheContext contains valid data.

        Raises:
            TypeError: If fields have wrong types
            ValueError: If tensors have invalid properties
            RuntimeError: If callables fail basic invocation tests

        This method should be called after creating a CacheContext to catch
        common developer errors early with clear error messages.
        """
        # Validate tensor fields
        if not isinstance(self.modulated_input, torch.Tensor):
            raise TypeError(f"modulated_input must be torch.Tensor, got {type(self.modulated_input)}")

        if not isinstance(self.hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be torch.Tensor, got {type(self.hidden_states)}")

        if self.encoder_hidden_states is not None and not isinstance(self.encoder_hidden_states, torch.Tensor):
            raise TypeError(
                f"encoder_hidden_states must be torch.Tensor or None, got {type(self.encoder_hidden_states)}"
            )

        if not isinstance(self.temb, torch.Tensor):
            raise TypeError(f"temb must be torch.Tensor, got {type(self.temb)}")

        # Validate callables
        if not callable(self.run_transformer_blocks):
            raise TypeError(f"run_transformer_blocks must be callable, got {type(self.run_transformer_blocks)}")

        if not callable(self.postprocess):
            raise TypeError(f"postprocess must be callable, got {type(self.postprocess)}")

        # Validate tensor shapes are compatible
        if self.modulated_input.shape[0] != self.hidden_states.shape[0]:
            raise ValueError(
                f"Batch size mismatch: modulated_input has batch size "
                f"{self.modulated_input.shape[0]}, but hidden_states has "
                f"{self.hidden_states.shape[0]}"
            )

        # Validate devices match
        if self.modulated_input.device != self.hidden_states.device:
            raise ValueError(
                f"Device mismatch: modulated_input on {self.modulated_input.device}, "
                f"hidden_states on {self.hidden_states.device}"
            )


def extract_bagel_context(
    module: nn.Module,
    x_t: torch.Tensor,
    timestep: torch.Tensor | float | int,
    packed_vae_token_indexes: torch.LongTensor,
    packed_vae_position_ids: torch.LongTensor,
    packed_text_ids: torch.LongTensor,
    packed_text_indexes: torch.LongTensor,
    packed_position_ids: torch.LongTensor,
    packed_seqlens: torch.IntTensor,
    past_key_values: Any,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for Bagel model.

    Args:
        module: Bagel instance
        x_t: Latent image input
        timestep: Current timestep
        packed_vae_token_indexes: Indexes for VAE tokens in packed sequence
        packed_vae_position_ids: Position IDs for VAE tokens
        packed_text_ids: Text token IDs
        packed_text_indexes: Indexes for text tokens in packed sequence
        packed_position_ids: Global position IDs
        packed_seqlens: Sequence lengths
        past_key_values: KV cache
        **kwargs: Additional keyword arguments

    Returns:
        CacheContext with all information needed for generic caching
    """

    # 1. Embed text
    packed_text_embedding = module.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), module.hidden_size))
    packed_sequence[packed_text_indexes] = packed_text_embedding

    # 2. Embed timestep
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], device=x_t.device)
    if timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)

    # 3. Embed image (x_t)
    packed_pos_embed = module.latent_pos_embed(packed_vae_position_ids)
    packed_timestep_embeds = module.time_embedder(timestep)

    x_t_emb = module.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
    if x_t_emb.dtype != packed_sequence.dtype:
        x_t_emb = x_t_emb.to(packed_sequence.dtype)

    packed_sequence[packed_vae_token_indexes] = x_t_emb

    # Use the full packed sequence as modulated input to match hidden_states size
    modulated_input = packed_sequence

    def run_transformer_blocks():
        extra_inputs = {}
        if module.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = module.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            past_key_values=past_key_values,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        return (output.packed_query_sequence,)

    def postprocess(h):
        v_t = module.llm2vae(h)
        v_t = v_t[packed_vae_token_indexes]
        return v_t

    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=packed_sequence,  # Use full packed sequence
        encoder_hidden_states=None,
        temb=packed_timestep_embeds,  # Approximate
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_zimage_context(
    module: nn.Module,
    x: list[torch.Tensor],
    t: torch.Tensor,
    cap_feats: list[torch.Tensor],
    patch_size: int = 2,
    f_patch_size: int = 1,
    **kwargs: Any,
) -> CacheContext:
    """Extract cache context for ZImageTransformer2DModel."""
    from torch.nn.utils.rnn import pad_sequence

    if not hasattr(module, "layers") or len(module.layers) == 0:
        raise ValueError("Module must have main transformer layers")

    bsz = len(x)
    device = x[0].device

    # ============================================================================
    # PREPROCESSING (Z-Image specific)
    # ============================================================================
    # Scale timestep and create timestep embedding
    t_scaled = t * module.t_scale
    adaln_input = module.t_embedder(t_scaled)

    # Patchify and embed inputs
    (
        x_patches,
        cap_feats_processed,
        x_size,
        x_pos_ids,
        cap_pos_ids,
        x_inner_pad_mask,
        cap_inner_pad_mask,
    ) = module.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

    # Process image patches through embedder and noise refiner
    x_item_seqlens = [len(_) for _ in x_patches]
    x_max_item_seqlen = max(x_item_seqlens)

    x_embedded = torch.cat(x_patches, dim=0)
    x_embedded = module.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_embedded)

    # Match adaln_input dtype to x_embedded
    adaln_input = adaln_input.type_as(x_embedded)

    # Apply pad token
    x_embedded[torch.cat(x_inner_pad_mask)] = module.x_pad_token
    x_list = list(x_embedded.split(x_item_seqlens, dim=0))

    # Compute rope embeddings for image patches
    x_cos, x_sin = module.rope_embedder(torch.cat(x_pos_ids, dim=0))
    x_cos = list(x_cos.split(x_item_seqlens, dim=0))
    x_sin = list(x_sin.split(x_item_seqlens, dim=0))

    # Pad sequences for batch processing
    x_batched = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    x_cos_batched = pad_sequence(x_cos, batch_first=True, padding_value=0.0)
    x_sin_batched = pad_sequence(x_sin, batch_first=True, padding_value=0.0)
    x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(x_item_seqlens):
        x_attn_mask[i, :seq_len] = 1

    # Run noise refiner blocks
    for layer in module.noise_refiner:
        x_batched = layer(x_batched, x_attn_mask, x_cos_batched, x_sin_batched, adaln_input)

    # Process caption features through embedder and context refiner
    cap_item_seqlens = [len(_) for _ in cap_feats_processed]
    cap_max_item_seqlen = max(cap_item_seqlens)

    cap_embedded = torch.cat(cap_feats_processed, dim=0)
    cap_embedded = module.cap_embedder(cap_embedded)
    cap_embedded[torch.cat(cap_inner_pad_mask)] = module.cap_pad_token
    cap_list = list(cap_embedded.split(cap_item_seqlens, dim=0))

    # Compute rope embeddings for caption
    cap_cos, cap_sin = module.rope_embedder(torch.cat(cap_pos_ids, dim=0))
    cap_cos = list(cap_cos.split(cap_item_seqlens, dim=0))
    cap_sin = list(cap_sin.split(cap_item_seqlens, dim=0))

    # Pad sequences for batch processing
    cap_batched = pad_sequence(cap_list, batch_first=True, padding_value=0.0)
    cap_cos_batched = pad_sequence(cap_cos, batch_first=True, padding_value=0.0)
    cap_sin_batched = pad_sequence(cap_sin, batch_first=True, padding_value=0.0)
    cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(cap_item_seqlens):
        cap_attn_mask[i, :seq_len] = 1

    # Run context refiner blocks
    for layer in module.context_refiner:
        cap_batched = layer(cap_batched, cap_attn_mask, cap_cos_batched, cap_sin_batched)

    # Create unified sequence (image + caption)
    unified_list = []
    unified_cos_list = []
    unified_sin_list = []
    for i in range(bsz):
        x_len = x_item_seqlens[i]
        cap_len = cap_item_seqlens[i]
        unified_list.append(torch.cat([x_batched[i][:x_len], cap_batched[i][:cap_len]]))
        unified_cos_list.append(torch.cat([x_cos_batched[i][:x_len], cap_cos_batched[i][:cap_len]]))
        unified_sin_list.append(torch.cat([x_sin_batched[i][:x_len], cap_sin_batched[i][:cap_len]]))

    unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
    unified_max_item_seqlen = max(unified_item_seqlens)

    unified = pad_sequence(unified_list, batch_first=True, padding_value=0.0)
    unified_cos = pad_sequence(unified_cos_list, batch_first=True, padding_value=0.0)
    unified_sin = pad_sequence(unified_sin_list, batch_first=True, padding_value=0.0)
    unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(unified_item_seqlens):
        unified_attn_mask[i, :seq_len] = 1

    # ============================================================================
    # EXTRACT MODULATED INPUT (for cache decision)
    # ============================================================================
    # Use the first main transformer block's modulation
    # The main layers have modulation=True and process the unified sequence
    block = module.layers[0]
    # Get modulation parameters: scale_msa, gate_msa, scale_mlp, gate_mlp
    mod_params = block.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
    scale_msa = 1.0 + mod_params[0]
    # Extract modulated input: normalized hidden states scaled by modulation
    modulated_input = block.attention_norm1(unified) * scale_msa

    # ============================================================================
    # DEFINE TRANSFORMER EXECUTION (Z-Image specific)
    # ============================================================================
    def run_transformer_blocks():
        """Execute all Z-Image main transformer blocks."""
        h = unified
        for layer in module.layers:
            h = layer(h, unified_attn_mask, unified_cos, unified_sin, adaln_input)
        return (h,)

    # ============================================================================
    # DEFINE POSTPROCESSING (Z-Image specific)
    # ============================================================================
    def postprocess(h):
        """Apply Z-Image specific output postprocessing."""
        h = module.all_final_layer[f"{patch_size}-{f_patch_size}"](h, adaln_input)
        h = list(h.unbind(dim=0))
        output = module.unpatchify(h, x_size, patch_size, f_patch_size)
        return output, {}

    # ============================================================================
    # RETURN CONTEXT
    # ============================================================================
    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=unified,
        encoder_hidden_states=None,  # Z-Image uses unified sequence, no separate encoder states
        temb=adaln_input,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
        extra_states={
            "unified_attn_mask": unified_attn_mask,
            "unified_cos": unified_cos,
            "unified_sin": unified_sin,
            "x_size": x_size,
            "x_item_seqlens": x_item_seqlens,
            "patch_size": patch_size,
            "f_patch_size": f_patch_size,
        },
    )


def extract_longcat_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    guidance: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
    **kwargs: Any,
) -> CacheContext:
    # TODO (Alex) - Refactor TeaCache extractors to more tightly integrate with .forward
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    # 1. Model specific preprocessing
    fwd_context = get_forward_context()
    sp_size = module.parallel_config.sequence_parallel_size
    if sp_size is not None and sp_size > 1:
        # NOTE: For now, we set this to False on the forward context
        # to be consistent with LongCat Image's current behavior when
        # TeaCache is enabled. We do not need to reset it in post process
        # since we should never split text embed in sp for this model.
        fwd_context.split_text_embed_in_sp = False

    hidden_states = module.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000

    temb = module.time_embed(timestep, hidden_states.dtype)
    encoder_hidden_states = module.context_embedder(encoder_hidden_states)

    # Compute RoPE embeddings via rope_preparer module
    # _sp_plan will automatically shard img_cos/img_sin (outputs 2, 3)
    # txt_cos/txt_sin (outputs 0, 1) remain replicated for dual-stream attention
    txt_cos, txt_sin, img_cos, img_sin = module.rope_preparer(txt_ids, img_ids)

    # Reconstruct image_rotary_emb with chunked values
    # Final shape: (txt_seq_len + img_seq_len // SP, head_dim)
    image_rotary_emb = (
        torch.cat([txt_cos, img_cos], dim=0),
        torch.cat([txt_sin, img_sin], dim=0),
    )

    # 2. Extract the modulated output from the first mm-DiT block
    first_block = module.transformer_blocks[0]
    img_modulated = first_block.norm1(hidden_states, emb=temb)[0]

    # 3. Define the transformer execution
    def run_transformer_blocks():
        """Execute all Longcat transformer blocks."""
        h = hidden_states
        e = encoder_hidden_states
        for block in module.transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        for block in module.single_transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        # Hook expects hidden states to be first
        return (h, e)

    # 4. Postprocessing
    def postprocess(h):
        """Apply Longcat-specific output postprocessing."""
        h = module.norm_out(h, temb)
        output = module.proj_out(h)
        return Transformer2DModelOutput(sample=output)

    # 5. Return the CacheContext
    return CacheContext(
        modulated_input=img_modulated,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


# Registry for model-specific extractors
# Key: Transformer class name
# Value: extractor function with signature (module, *args, **kwargs) -> CacheContext
#
# Note: Use the transformer class name as specified in pipelines as TeaCache hooks operate
# on the transformer module and multiple pipelines can share the same transformer.
EXTRACTOR_REGISTRY: dict[str, Callable] = {
    "Bagel": extract_bagel_context,
    "LongCatImageTransformer2DModel": extract_longcat_context,
    # Future models:
    # "CogVideoXTransformer3DModel": extract_cogvideox_context,
}


def register_extractor(transformer_cls_name: str, extractor_fn: Callable) -> None:
    """
    Register a new extractor function for a model type.

    This allows extending TeaCache support to new models without modifying
    the core TeaCache code.

    Args:
        transformer_cls_name: Transformer model type identifier (class name or type string)
        extractor_fn: Function with signature (module, *args, **kwargs) -> CacheContext

    Example:
        >>> def extract_flux_context(module, hidden_states, timestep, guidance=None, **kwargs):
        ...     # Preprocessing
        ...     temb = module.time_text_embed(timestep, guidance)
        ...     # Extract modulated input
        ...     modulated = module.transformer_blocks[0].norm1(hidden_states, emb=temb)
        ...     # Define execution
        ...     def run_blocks():
        ...         h = hidden_states
        ...         for block in module.transformer_blocks:
        ...             h = block(h, temb=temb)
        ...         return (h,)
        ...     # Define postprocessing
        ...     def postprocess(h):
        ...         return module.proj_out(module.norm_out(h, temb))
        ...     # Return context
        ...     return CacheContext(modulated, hidden_states, None, temb, run_blocks, postprocess)
        >>> register_extractor("FluxTransformer2DModel", extract_flux_context)
    """
    EXTRACTOR_REGISTRY[transformer_cls_name] = extractor_fn


def get_extractor(transformer_cls_name: str) -> Callable:
    """
    Get extractor function for given transformer class.

    This function looks up the extractor based on the exact transformer_cls_name string,
    which should match the transformer type in the pipeline (i.e., pipeline.transformer.__class__.__name__).

    Args:
        transformer_cls_name: Transformer class name (e.g., "QwenImageTransformer2DModel")
                                Must exactly match a key in EXTRACTOR_REGISTRY.

    Returns:
        Extractor function with signature (module, *args, **kwargs) -> CacheContext

    Raises:
        ValueError: If model type not found in registry

    Example:
        >>> # Get extractor for QwenImageTransformer2DModel
        >>> extractor = get_extractor("QwenImageTransformer2DModel")
        >>> ctx = extractor(transformer, hidden_states, encoder_hidden_states, timestep, ...)
    """
    # Direct lookup - no substring matching
    if transformer_cls_name in EXTRACTOR_REGISTRY:
        return EXTRACTOR_REGISTRY[transformer_cls_name]

    # No match found
    available_types = list(EXTRACTOR_REGISTRY.keys())
    raise ValueError(
        f"Unknown model type: '{transformer_cls_name}'. "
        f"Available types: {available_types}\n"
        f"To add support for a new model, use register_extractor() or add to EXTRACTOR_REGISTRY."
    )
