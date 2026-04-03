# config_ming_tts.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from transformers import PretrainedConfig, Qwen2Config

from .audio_tokenizer.configuration_audio_vae import AudioVAEconfig

# ---------------------------------------------------------------------------
# Token IDs (confirmed from tokenizer_config.json)
# ---------------------------------------------------------------------------

AUDIO_DUMMY_TOKEN_ID: int = 151705  # <audioPatch>
AUDIO_START_TOKEN_ID: int = 151706  # <audio>
AUDIO_END_TOKEN_ID: int = 151707  # </audio>
AUDIO_EOS_TOKEN_ID: int = 151704  # <end_of_audio>
VISION_START_TOKEN_ID: int = 151652  # <|vision_start|>

TEXT_EOS_TOKEN_ID: int = 151669  # <text_eos>
PAD_TOKEN_ID: int = 151643  # <|endoftext|>

# Backward-compat alias for older code paths
EOS_TOKEN_ID: int = TEXT_EOS_TOKEN_ID


# ---------------------------------------------------------------------------
# Architectural constants (confirmed from original config.json)
# ---------------------------------------------------------------------------

LATENT_DIM: int = 64
PATCH_SIZE: int = 4
HISTORY_PATCH_SIZE: int = 32
LLM_HIDDEN_SIZE: int = 896
LLM_VOCAB_SIZE: int = 151936
AGGREGATOR_HIDDEN_SIZE: int = 1024
VAE_PATCH_SIZE: int = 4
SAMPLE_RATE: int = 44100

# AudioVAE frame/hop geometry (confirmed)
AUDIO_FRAME_HOP: int = 882  # enc input_dim / hop_size / dec output_dim

# stop_head defaults
STOP_HEAD_MIN_STEPS: int = 3
STOP_HEAD_THRESHOLD: float = 0.5

# FlowLoss sampling defaults
DEFAULT_CFG: float = 2.0
DEFAULT_SIGMA: float = 0.25
DEFAULT_TEMPERATURE: float = 0.0

# Connector / Stage-2 streaming defaults (runtime tuning)
LATENT_CHUNK_SIZE: int = 10
LATENT_LEFT_CONTEXT: int = 0
MAX_DECODE_STEPS: int = 200

# seq_data.extra_data keys
KEY_LATENT_HISTORY: str = "ming_latent_history"
KEY_DECODE_STEP: str = "ming_decode_step"
KEY_LAST_STOP_PROB: str = "ming_last_stop_prob"
KEY_NEXT_EMBEDS: str = "ming_next_embeds"
KEY_PROMPT_LATENTS: str = "ming_prompt_latents"
KEY_PROMPT_LATENT_TAIL: str = "ming_prompt_latent_tail"
KEY_SPEAKER_EMBEDDING: str = "ming_speaker_embedding"
KEY_REQUEST_ID: str = "ming_request_id"
KEY_CHUNK_ID: str = "ming_chunk_id"
KEY_CFG: str = "ming_cfg"
KEY_SIGMA: str = "ming_sigma"
KEY_TEMPERATURE: str = "ming_temperature"
KEY_MAX_DECODE_STEPS: str = "ming_max_decode_steps"
KEY_MIN_DECODE_STEPS: str = "ming_min_decode_steps"
KEY_TEXT_MODE: str = "ming_text_mode"


@dataclass
class MingTTSConfig:
    """Flat config object shared by Stage-1 and Stage-2. Build via from_hf_config()."""

    # --- LLM backbone ---
    llm_hidden_size: int = LLM_HIDDEN_SIZE
    llm_vocab_size: int = LLM_VOCAB_SIZE
    llm_config: dict[str, Any] = field(default_factory=dict)

    # --- Audio latent space ---
    latent_dim: int = LATENT_DIM
    patch_size: int = PATCH_SIZE
    history_patch_size: int = HISTORY_PATCH_SIZE

    # --- Flow / Aggregator sub-configs ---
    ditar_config: dict[str, Any] = field(default_factory=dict)
    aggregator_config: dict[str, Any] = field(default_factory=dict)

    # --- AudioVAE ---
    audio_tokenizer_config: AudioVAEconfig | None = None
    vae_patch_size: int = VAE_PATCH_SIZE
    sample_rate: int = SAMPLE_RATE
    audio_frame_hop: int = AUDIO_FRAME_HOP

    # --- Generation control ---
    cfg: float = DEFAULT_CFG
    sigma: float = DEFAULT_SIGMA
    temperature: float = DEFAULT_TEMPERATURE
    stop_head_min_steps: int = STOP_HEAD_MIN_STEPS
    stop_head_threshold: float = STOP_HEAD_THRESHOLD
    max_decode_steps: int = MAX_DECODE_STEPS

    # --- Stage-2 chunking (runtime tuning) ---
    latent_chunk_size: int = LATENT_CHUNK_SIZE
    latent_left_context: int = LATENT_LEFT_CONTEXT

    # --- Token IDs ---
    text_eos_token_id: int = TEXT_EOS_TOKEN_ID
    eos_token_id: int = TEXT_EOS_TOKEN_ID  # compat alias
    pad_token_id: int = PAD_TOKEN_ID
    audio_dummy_token_id: int = AUDIO_DUMMY_TOKEN_ID
    audio_start_token_id: int = AUDIO_START_TOKEN_ID
    audio_end_token_id: int = AUDIO_END_TOKEN_ID
    audio_eos_token_id: int = AUDIO_EOS_TOKEN_ID

    @classmethod
    def from_hf_config(cls, hf_config: PretrainedConfig) -> MingTTSConfig:
        """
        Build from vllm-omni's hf_config. Supports nested configs as objects or dicts.
        """

        # --- Read nested sub-configs (must NOT read flat hf_config attrs for these) ---
        llm_raw = getattr(hf_config, "llm_config", {}) or {}
        ditar_raw = getattr(hf_config, "ditar_config", {}) or {}
        agg_raw = getattr(hf_config, "aggregator_config", {}) or {}
        atc_raw = getattr(hf_config, "audio_tokenizer_config", None)

        llm_dict = _to_plain_dict(llm_raw)
        ditar = _to_plain_dict(ditar_raw)
        agg = _to_plain_dict(agg_raw)

        # Keep Ming DiT backend explicit; original checkpoint uses "torch"
        ditar.setdefault("attn_backend", "torch")

        atc = _coerce_audio_vae_config(atc_raw)

        # --- Pull nested values safely ---
        atc_enc_latent_dim = _nested_get(atc, "enc_kwargs", "latent_dim", default=LATENT_DIM)
        atc_patch_size = _nested_get(atc, "patch_size", default=VAE_PATCH_SIZE)
        atc_sample_rate = _nested_get(atc, "sample_rate", default=SAMPLE_RATE)

        enc_input_dim = _nested_get(atc, "enc_kwargs", "input_dim", default=AUDIO_FRAME_HOP)
        enc_hop_size = _nested_get(atc, "enc_kwargs", "hop_size", default=AUDIO_FRAME_HOP)
        dec_output_dim = _nested_get(atc, "dec_kwargs", "output_dim", default=AUDIO_FRAME_HOP)

        cfg = cls(
            llm_hidden_size=llm_dict.get("hidden_size", LLM_HIDDEN_SIZE),
            llm_vocab_size=llm_dict.get("vocab_size", LLM_VOCAB_SIZE),
            llm_config=llm_dict,
            latent_dim=atc_enc_latent_dim,
            patch_size=ditar.get("patch_size", PATCH_SIZE),
            history_patch_size=ditar.get("history_patch_size", HISTORY_PATCH_SIZE),
            ditar_config=ditar,
            aggregator_config=agg,
            audio_tokenizer_config=atc,
            vae_patch_size=atc_patch_size,
            sample_rate=atc_sample_rate,
            audio_frame_hop=enc_hop_size if enc_hop_size is not None else AUDIO_FRAME_HOP,
        )

        # Optional debug cache (safe to keep)
        cfg._enc_input_dim = enc_input_dim
        cfg._enc_hop_size = enc_hop_size
        cfg._dec_output_dim = dec_output_dim

        return cfg

    def validate(self) -> None:
        """Run before GPU allocation/weight loading. Raises ValueError on mismatches."""

        # --- Token IDs ---
        if self.audio_dummy_token_id != 151705:
            raise ValueError(
                f"audio_dummy_token_id={self.audio_dummy_token_id}, expected 151705 (<audioPatch>). "
                "Wrong tokenizer/checkpoint?"
            )
        if self.audio_eos_token_id != 151704:
            raise ValueError(
                f"audio_eos_token_id={self.audio_eos_token_id}, expected 151704 (<end_of_audio>). "
                "Wrong tokenizer/checkpoint?"
            )
        if self.text_eos_token_id != 151669:
            raise ValueError(
                f"text_eos_token_id={self.text_eos_token_id}, expected 151669 (<text_eos>). Wrong tokenizer/checkpoint?"
            )

        # --- Required sub-config ---
        if self.audio_tokenizer_config is None:
            raise ValueError("audio_tokenizer_config is None. Nested AudioVAE config was not deserialized correctly.")

        # --- Confirmed checkpoint-family constants ---
        if self.latent_dim != LATENT_DIM:
            raise ValueError(
                f"latent_dim mismatch: got {self.latent_dim}, expected {LATENT_DIM}. "
                "Check audio_tokenizer_config.enc_kwargs.latent_dim."
            )
        if self.patch_size != PATCH_SIZE:
            raise ValueError(
                f"patch_size mismatch: got {self.patch_size}, expected {PATCH_SIZE}. Check ditar_config.patch_size."
            )
        if self.history_patch_size != HISTORY_PATCH_SIZE:
            raise ValueError(
                f"history_patch_size mismatch: got {self.history_patch_size}, expected {HISTORY_PATCH_SIZE}. "
                "Check ditar_config.history_patch_size."
            )
        if self.llm_hidden_size != LLM_HIDDEN_SIZE:
            raise ValueError(
                f"llm_hidden_size mismatch: got {self.llm_hidden_size}, expected {LLM_HIDDEN_SIZE}. "
                "Check llm_config.hidden_size."
            )
        if self.llm_vocab_size != LLM_VOCAB_SIZE:
            raise ValueError(f"llm_vocab_size mismatch: got {self.llm_vocab_size}, expected {LLM_VOCAB_SIZE}.")
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"sample_rate mismatch: got {self.sample_rate}, expected {SAMPLE_RATE}.")

        # --- Cross-config consistency checks ---
        if self.vae_patch_size != self.patch_size:
            raise ValueError(f"VAE patch size ({self.vae_patch_size}) != flow/DiT patch size ({self.patch_size}).")

        llm_hidden_from_cfg = self.llm_config.get("hidden_size")
        if llm_hidden_from_cfg is not None and llm_hidden_from_cfg != self.llm_hidden_size:
            raise ValueError(
                f"llm_hidden_size ({self.llm_hidden_size}) != llm_config.hidden_size ({llm_hidden_from_cfg})."
            )

        agg_h = self.aggregator_config.get("hidden_size")
        dit_h = self.ditar_config.get("hidden_size")
        if agg_h is not None and dit_h is not None and agg_h != dit_h:
            raise ValueError(f"aggregator_config.hidden_size ({agg_h}) != ditar_config.hidden_size ({dit_h}).")
        if agg_h is not None and agg_h != AGGREGATOR_HIDDEN_SIZE:
            raise ValueError(f"aggregator hidden_size mismatch: got {agg_h}, expected {AGGREGATOR_HIDDEN_SIZE}.")
        if dit_h is not None and dit_h != AGGREGATOR_HIDDEN_SIZE:
            raise ValueError(f"ditar hidden_size mismatch: got {dit_h}, expected {AGGREGATOR_HIDDEN_SIZE}.")

        atc = self.audio_tokenizer_config
        enc_latent = _nested_get(atc, "enc_kwargs", "latent_dim", default=None)
        dec_latent = _nested_get(atc, "dec_kwargs", "latent_dim", default=None)
        if enc_latent is not None and enc_latent != self.latent_dim:
            raise ValueError(f"audio enc latent_dim ({enc_latent}) != Ming latent_dim ({self.latent_dim}).")
        if dec_latent is not None and dec_latent != self.latent_dim:
            raise ValueError(f"audio dec latent_dim ({dec_latent}) != Ming latent_dim ({self.latent_dim}).")

        atc_patch = _nested_get(atc, "patch_size", default=None)
        if atc_patch is not None and atc_patch != self.vae_patch_size:
            raise ValueError(
                f"audio_tokenizer_config.patch_size ({atc_patch}) != vae_patch_size ({self.vae_patch_size})."
            )

        atc_sr = _nested_get(atc, "sample_rate", default=None)
        if atc_sr is not None and atc_sr != self.sample_rate:
            raise ValueError(f"audio_tokenizer_config.sample_rate ({atc_sr}) != sample_rate ({self.sample_rate}).")

        enc_input_dim = _nested_get(atc, "enc_kwargs", "input_dim", default=None)
        enc_hop_size = _nested_get(atc, "enc_kwargs", "hop_size", default=None)
        dec_output_dim = _nested_get(atc, "dec_kwargs", "output_dim", default=None)

        if enc_input_dim is not None and enc_hop_size is not None and enc_input_dim != enc_hop_size:
            raise ValueError(f"AudioVAE encoder input_dim ({enc_input_dim}) != hop_size ({enc_hop_size}).")
        if enc_hop_size is not None and dec_output_dim is not None and enc_hop_size != dec_output_dim:
            raise ValueError(
                f"AudioVAE encoder hop_size ({enc_hop_size}) != decoder output_dim ({dec_output_dim}). "
                "Expected 882 in this checkpoint family."
            )

        # Runtime tuning sanity
        if self.latent_chunk_size <= 0:
            raise ValueError(f"latent_chunk_size must be > 0, got {self.latent_chunk_size}.")
        if self.latent_left_context < 0:
            raise ValueError(f"latent_left_context must be >= 0, got {self.latent_left_context}.")
        if self.max_decode_steps <= 0:
            raise ValueError(f"max_decode_steps must be > 0, got {self.max_decode_steps}.")
        if not (0.0 <= self.stop_head_threshold <= 1.0):
            raise ValueError(f"stop_head_threshold must be in [0,1], got {self.stop_head_threshold}.")
        if self.stop_head_min_steps < 0:
            raise ValueError(f"stop_head_min_steps must be >= 0, got {self.stop_head_min_steps}.")

    def make_qwen2_config(self) -> Qwen2Config:
        """Reconstruct Qwen2Config for Stage-1 LLM backbone init."""
        if not self.llm_config:
            raise ValueError("llm_config is empty; from_hf_config() failed to parse nested llm_config.")
        return Qwen2Config.from_dict(self.llm_config)

    @property
    def latent_patch_shape(self) -> tuple[int, int]:
        return (self.patch_size, self.latent_dim)

    @property
    def chunk_frames(self) -> int:
        return self.latent_chunk_size * self.patch_size

    @property
    def approx_chunk_seconds(self) -> float:
        # One latent frame ~ one 882-sample hop in this checkpoint family.
        return (self.chunk_frames * self.audio_frame_hop) / float(self.sample_rate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """Normalize nested config objects into plain dicts when possible."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, PretrainedConfig):
        return obj.to_dict()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return dict(obj.to_dict())
        except Exception:
            pass
    try:
        return dict(vars(obj))
    except Exception:
        return {}


def _coerce_audio_vae_config(atc_raw: Any) -> AudioVAEconfig | None:
    """
    Normalize audio_tokenizer_config into AudioVAEconfig when possible.
    Handles:
      - already AudioVAEconfig
      - dict
      - PretrainedConfig-like object
    """
    if atc_raw is None:
        return None
    atc_dict = _to_plain_dict(atc_raw)
    if not atc_dict:
        # Return raw object as fallback; _nested_get/validate can still work
        return atc_raw  # type: ignore[return-value]

    _normalize_audio_vae_backbone_attention(atc_dict)

    if hasattr(AudioVAEconfig, "from_dict") and callable(getattr(AudioVAEconfig, "from_dict")):
        try:
            return AudioVAEconfig.from_dict(atc_dict)  # type: ignore[misc]
        except Exception:
            pass
    try:
        return AudioVAEconfig(**atc_dict)  # type: ignore[arg-type]
    except Exception:
        return atc_raw  # type: ignore[return-value]


def _normalize_audio_vae_backbone_attention(atc_dict: dict[str, Any]) -> None:
    for branch in ("enc_kwargs", "dec_kwargs"):
        branch_cfg = atc_dict.get(branch)
        if not isinstance(branch_cfg, dict):
            continue
        backbone = branch_cfg.get("backbone")
        if not isinstance(backbone, dict):
            continue

        # Ming local validation should run on plain Transformers Qwen2Model
        # without requiring flash_attn. Force SDPA/eager-compatible settings.
        if backbone.get("_attn_implementation") == "flash_attention_2":
            backbone["_attn_implementation"] = "sdpa"
        if backbone.get("attn_implementation") in (None, "flash_attention_2"):
            backbone["attn_implementation"] = "sdpa"


def _nested_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safe nested attribute/key access for dicts and config-like objects."""
    cur = obj
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            cur = getattr(cur, k, None)
    return cur if cur is not None else default
