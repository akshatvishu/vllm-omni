# audio_tokenizer/modeling_audio_vae.py
import torch
import torch.nn as nn

from .configuration_audio_vae import AudioVAEconfig
from .vae_modules import Decoder, Encoder


class AudioVAE(nn.Module):
    def __init__(self, config: AudioVAEconfig):
        super().__init__()
        self.config = config

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

        # Semantic module is null for this checkpoint.
        if config.semantic_module_kwargs is not None:
            from .audio_encoder import WhisperAudioEncoder

            semantic_model = WhisperAudioEncoder.from_pretrained(dims=config.semantic_module_kwargs["whisper_encoder"])
        else:
            semantic_model = None

        self.decoder = Decoder(
            decoder_args=dec_kwargs["backbone"],  # IMPORTANT: decoder uses dec_kwargs.backbone
            output_dim=dec_kwargs["output_dim"],  # Ming checkpoint uses 882
            latent_dim=dec_kwargs["latent_dim"],
            semantic_model=semantic_model,
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
