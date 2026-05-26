# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/audio_tokenizer/modeling_audio_vae.py
# audio_tokenizer/modeling_audio_vae.py
import torch
import torch.nn as nn
from vllm.logger import init_logger

from .configuration_audio_vae import AudioVAEconfig
from .vae_modules import Decoder, Encoder

logger = init_logger(__name__)


def _get_backbone(config: AudioVAEconfig, branch: str):
    branch_cfg = getattr(config, branch, None)
    if not isinstance(branch_cfg, dict):
        return None
    backbone = branch_cfg.get("backbone")
    if not isinstance(backbone, dict):
        return None
    return backbone


def _maybe_fallback_attention(config: AudioVAEconfig) -> None:
    enc_backbone = _get_backbone(config, "enc_kwargs")
    dec_backbone = _get_backbone(config, "dec_kwargs")
    requested_attn_impl = "flash_attention_2"

    if dec_backbone is not None:
        requested_attn_impl = dec_backbone.get(
            "_attn_implementation",
            dec_backbone.get("attn_implementation", requested_attn_impl),
        )
    elif enc_backbone is not None:
        requested_attn_impl = enc_backbone.get(
            "_attn_implementation",
            enc_backbone.get("attn_implementation", requested_attn_impl),
        )

    if requested_attn_impl != "flash_attention_2":
        return

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        if enc_backbone is not None:
            enc_backbone["_attn_implementation"] = "sdpa"
            enc_backbone["attn_implementation"] = "sdpa"
        if dec_backbone is not None:
            dec_backbone["_attn_implementation"] = "sdpa"
            dec_backbone["attn_implementation"] = "sdpa"
        logger.warning("flash_attn not available, falling back to sdpa for Ming audio VAE")


class AudioVAE(nn.Module):
    def __init__(self, config: AudioVAEconfig):
        super().__init__()
        self.config = config
        _maybe_fallback_attention(self.config)

        # --- Ming/Bailing config sanity (fail early on bad nested config parsing) ---
        enc_kwargs = config.enc_kwargs
        dec_kwargs = config.dec_kwargs

        # Required nested fields
        for k in ("backbone", "input_dim", "latent_dim"):
            if k not in enc_kwargs:
                raise ValueError(f"AudioVAE.enc_kwargs missing required key: {k}")
        for k in ("backbone", "output_dim", "latent_dim"):
            if k not in dec_kwargs:
                raise ValueError(f"AudioVAE.dec_kwargs missing required key: {k}")

        # Ming-specific geometry checks (safe because this integration targets Ming checkpoint family)
        hop_size = enc_kwargs.get("hop_size", enc_kwargs["input_dim"])
        if enc_kwargs["input_dim"] != hop_size:
            raise ValueError(f"AudioVAE encoder input_dim ({enc_kwargs['input_dim']}) != hop_size ({hop_size}).")
        if hop_size != dec_kwargs["output_dim"]:
            raise ValueError(
                f"AudioVAE encoder hop_size ({hop_size}) != decoder output_dim ({dec_kwargs['output_dim']})."
            )

        self.encoder = Encoder(
            encoder_args=enc_kwargs["backbone"],
            input_dim=enc_kwargs["input_dim"],
            hop_size=hop_size,
            latent_dim=enc_kwargs["latent_dim"],
            patch_size=config.patch_size,
        )

        if config.semantic_module_kwargs is not None:
            raise ValueError("Ming dense 0.5B expects semantic_module_kwargs to be null.")

        self.decoder = Decoder(
            decoder_args=dec_kwargs["backbone"],  # IMPORTANT: decoder uses dec_kwargs.backbone
            output_dim=dec_kwargs["output_dim"],  # Ming checkpoint uses 882
            latent_dim=dec_kwargs["latent_dim"],
            patch_size=config.patch_size,
        )

    @torch.inference_mode()
    def encode_latent(
        self,
        waveform: torch.Tensor,
        waveform_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode waveform -> acoustic latent.
        """
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform rank-2 [Batch, Time], got {tuple(waveform.shape)}")
        if waveform_length.ndim != 1:
            raise ValueError(f"Expected waveform_length rank-1 [Batch], got {tuple(waveform_length.shape)}")
        if waveform.shape[0] != waveform_length.shape[0]:
            raise ValueError(
                "Batch mismatch: "
                f"waveform batch={waveform.shape[0]} vs "
                f"waveform_length batch={waveform_length.shape[0]}"
            )
        if torch.any(waveform_length <= 0):
            raise ValueError("waveform_length must be strictly positive.")

        frame_num = torch.ceil(waveform_length / self.config.enc_kwargs["input_dim"]).to(torch.int32)
        if self.config.patch_size != -1:
            frame_num = torch.ceil(frame_num / self.config.patch_size)

        h, _ = self.encoder(waveform)
        h = h.transpose(1, 2)  # [B, 2*latent_dim, T] (posterior params: mean + logvar)

        # Inline OobleckDiagonalGaussianDistribution.sample()
        mean, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        latent = mean + std * torch.randn_like(mean)  # [B, latent_dim, T]
        latent = latent.transpose(1, 2)  # [B, T, d/2]

        return latent, frame_num

    @torch.inference_mode()
    def decode(
        self,
        latent: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
        stream_state: tuple = (None, None, None),
        last_chunk: bool = False,
    ) -> tuple[torch.Tensor, tuple, object]:
        """
        Decode acoustic latent -> waveform.
        """
        if latent.dim() != 3:
            raise ValueError(f"Expected latent rank-3 [B,T,D], got shape={tuple(latent.shape)}")
        if latent.shape[0] <= 0:
            raise ValueError("latent batch size must be positive.")

        target_dtype = next(self.decoder.parameters()).dtype
        target_device = next(self.decoder.parameters()).device
        if latent.dtype != target_dtype or latent.device != target_device:
            latent = latent.to(device=target_device, dtype=target_dtype)

        expected_latent_dim = self.config.dec_kwargs["latent_dim"]
        if latent.shape[-1] != expected_latent_dim:
            raise ValueError(f"Latent dim mismatch in decode(): got {latent.shape[-1]}, expected {expected_latent_dim}")

        waveform, stream_state, past_key_values = self.decoder.low_level_reconstruct(
            latent,
            past_key_values=past_key_values,
            use_cache=use_cache,
            stream_state=stream_state,
            last_chunk=last_chunk,
        )
        return waveform, stream_state, past_key_values
