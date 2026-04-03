from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .audio_tokenizer.modeling_audio_vae import AudioVAE
from .config_ming_tts import (
    AUDIO_DUMMY_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    KEY_PROMPT_LATENTS,
    KEY_SPEAKER_EMBEDDING,
    KEY_TEXT_MODE,
    LATENT_DIM,
    PATCH_SIZE,
    MingTTSConfig,
)
from .configuration_ming_dense import MingDenseConfig
from .prompt_builder import (
    build_dense_prompt_token_ids,
    coerce_speaker_embeddings,
    count_prompt_latent_patches,
    create_instruction,
    pad_prompt_waveform,
)


def _resolve_model_to_local_path(model: str) -> str:
    if os.path.isdir(model):
        return model

    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Ming ingress processor requires a local model snapshot, got {model!r}. "
            "Download the model first or pass a local path."
        ) from exc


def encode_prompt_waveform_to_frame_latents(
    encoder: AudioVAE,
    waveform: Any,
    waveform_length: Any = None,
    *,
    patch_size: int = PATCH_SIZE,
    latent_dim: int = LATENT_DIM,
    sample_rate: int,
    frame_hop: int,
) -> torch.Tensor:
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = pad_prompt_waveform(
        waveform,
        patch_size=patch_size,
        sample_rate=sample_rate,
        frame_hop=frame_hop,
    )
    if waveform_length is None:
        waveform_length = torch.full(
            (waveform.shape[0],),
            waveform.shape[-1],
            dtype=torch.int32,
            device=waveform.device,
        )
    elif not isinstance(waveform_length, torch.Tensor):
        waveform_length = torch.as_tensor(waveform_length, dtype=torch.int32, device=waveform.device)
    else:
        waveform_length = waveform_length.to(device=waveform.device, dtype=torch.int32)

    latent, _ = encoder.encode_latent(waveform, waveform_length)
    if latent.ndim == 3 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    count_prompt_latent_patches(
        latent,
        patch_size=patch_size,
        latent_dim=latent_dim,
    )
    return latent.detach().to("cpu", dtype=torch.float32).contiguous()


def _iter_model_safetensors(local_model_path: str) -> list[Path]:
    model_root = Path(local_model_path)
    index_path = model_root / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        filenames = sorted(set(index_data.get("weight_map", {}).values()))
        if not filenames:
            raise RuntimeError(f"No checkpoint shards listed in {index_path}")
        return [model_root / filename for filename in filenames]

    single_file = model_root / "model.safetensors"
    if single_file.exists():
        return [single_file]

    files = sorted(model_root.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No .safetensors checkpoint found under {local_model_path}")
    return files


def _rebuild_prompt_token_ids_with_exact_patch_count(prompt_token_ids: Any, prompt_patch_count: int) -> list[int]:
    if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
        raise ValueError("Ming prompt finalization requires existing prompt_token_ids")

    audio_start_index = -1
    for idx in range(len(prompt_token_ids) - 1, -1, -1):
        if int(prompt_token_ids[idx]) == AUDIO_START_TOKEN_ID:
            audio_start_index = idx
            break
    if audio_start_index < 0:
        raise ValueError("Ming prompt finalization could not locate <audio> token")

    trailing_tokens = prompt_token_ids[audio_start_index + 1 :]
    if any(int(token_id) != AUDIO_DUMMY_TOKEN_ID for token_id in trailing_tokens):
        raise ValueError("Ming prompt finalization expected only trailing <audioPatch> tokens after <audio>")

    return prompt_token_ids[: audio_start_index + 1] + ([AUDIO_DUMMY_TOKEN_ID] * int(prompt_patch_count))


class MingIngressProcessor:
    def __init__(self, *, vllm_config: Any, tokenizer: Any):
        if tokenizer is None:
            raise RuntimeError("Ming ingress processor requires an initialized tokenizer")

        self.tokenizer = tokenizer
        self.model_path = _resolve_model_to_local_path(str(vllm_config.model_config.model))

        self.ming_config = MingTTSConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()
        self._audio_encoder = None

    def _load_prompt_audio_encoder(self) -> AudioVAE:
        if self._audio_encoder is not None:
            return self._audio_encoder

        config = MingDenseConfig.from_pretrained(self.model_path)
        if config.audio_tokenizer_config is None:
            raise RuntimeError("audio_tokenizer_config is missing; cannot build prompt audio encoder.")
        encoder = AudioVAE(config.audio_tokenizer_config).eval()
        state_dict = encoder.state_dict()
        loaded = 0
        with torch.no_grad():
            for shard_path in _iter_model_safetensors(self.model_path):
                with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if not key.startswith("audio."):
                            continue
                        name = key[len("audio.") :]
                        if name not in state_dict:
                            continue
                        target = state_dict[name]
                        target.copy_(handle.get_tensor(key).to(device=target.device, dtype=target.dtype))
                        loaded += 1
        if loaded == 0:
            raise RuntimeError("Prompt audio encoder received no audio.* weights from checkpoint.")

        self._audio_encoder = encoder
        return encoder

    def _encode_prompt_latents(self, prompt_waveform: Any, prompt_waveform_length: Any = None) -> torch.Tensor:
        return encode_prompt_waveform_to_frame_latents(
            self._load_prompt_audio_encoder(),
            prompt_waveform,
            prompt_waveform_length,
            patch_size=self.ming_config.patch_size,
            latent_dim=self.ming_config.latent_dim,
            sample_rate=self.ming_config.sample_rate,
            frame_hop=self.ming_config.audio_frame_hop,
        )

    def __call__(self, prompt: Any) -> Any:
        if not isinstance(prompt, dict):
            return prompt

        raw_additional_information = prompt.get("additional_information")
        if raw_additional_information is None:
            additional_information = {}
        elif isinstance(raw_additional_information, dict):
            additional_information = raw_additional_information
        else:
            return prompt

        modalities = prompt.get("modalities")
        text_mode = isinstance(modalities, (list, tuple)) and ("text" in modalities) and ("audio" not in modalities)
        if text_mode:
            finalized_prompt = copy.copy(prompt)
            finalized_additional_information = dict(additional_information)
            finalized_additional_information[KEY_TEXT_MODE] = True
            prompt_token_ids = finalized_prompt.get("prompt_token_ids")
            if isinstance(prompt_token_ids, list) and prompt_token_ids:
                if int(prompt_token_ids[-1]) == AUDIO_START_TOKEN_ID:
                    finalized_prompt["prompt_token_ids"] = prompt_token_ids[:-1]
            finalized_prompt["additional_information"] = finalized_additional_information
            return finalized_prompt

        prompt_waveform = additional_information.get("prompt_waveform", prompt.get("prompt_waveform"))
        prompt_text = additional_information.get("prompt_text", prompt.get("prompt_text"))
        if prompt_waveform is None or prompt_text is None:
            return prompt

        prompt_latents = additional_information.get(KEY_PROMPT_LATENTS, prompt.get("prompt_latents"))
        if prompt_latents is not None:
            raise ValueError(
                "Ming waveform cloning request provided both raw prompt_waveform and explicit prompt_latents. "
                "Choose exactly one source of truth."
            )

        prompt_waveform_length = additional_information.get("prompt_waveform_length", prompt.get("prompt_waveform_length"))
        prompt_latents = self._encode_prompt_latents(prompt_waveform, prompt_waveform_length)
        prompt_patch_count = count_prompt_latent_patches(
            prompt_latents,
            patch_size=self.ming_config.patch_size,
            latent_dim=self.ming_config.latent_dim,
        )

        finalized_prompt = copy.copy(prompt)
        finalized_additional_information = dict(additional_information)
        # Ingress owns raw waveform -> prompt latents. After this point Stage-0
        # should consume only finalized prompt latents, not the original waveform.
        finalized_additional_information.pop("prompt_waveform", None)
        finalized_additional_information.pop("prompt_waveform_length", None)
        finalized_additional_information.pop("prompt_waveforms", None)
        finalized_additional_information[KEY_PROMPT_LATENTS] = prompt_latents
        finalized_prompt["additional_information"] = finalized_additional_information
        finalized_prompt.pop("prompt_waveform", None)
        finalized_prompt.pop("prompt_waveform_length", None)
        finalized_prompt.pop("prompt_waveforms", None)

        prompt_prefix = finalized_prompt.get("prompt")
        text = finalized_prompt.get("text")
        if isinstance(prompt_prefix, str) and isinstance(text, str):
            speaker_embedding = finalized_prompt.get("speaker_embedding")
            if speaker_embedding is None:
                speaker_embedding = finalized_additional_information.get(KEY_SPEAKER_EMBEDDING)
            speaker_embeddings = coerce_speaker_embeddings(
                speaker_embedding,
                use_zero_spk_emb=bool(finalized_additional_information.get("use_zero_spk_emb", False)),
            )

            instruction = finalized_prompt.get("instruction")
            if instruction is None:
                instruction = finalized_additional_information.get("instruction")
            instruction_text = instruction if isinstance(instruction, str) else create_instruction(instruction)

            finalized_prompt["prompt_token_ids"] = build_dense_prompt_token_ids(
                self.tokenizer,
                prompt=prompt_prefix,
                text=text,
                instruction=instruction_text,
                prompt_text=prompt_text,
                speaker_count=0 if speaker_embeddings is None else len(speaker_embeddings),
                prompt_patch_count=prompt_patch_count,
            )
            return finalized_prompt

        finalized_prompt["prompt_token_ids"] = _rebuild_prompt_token_ids_with_exact_patch_count(
            finalized_prompt.get("prompt_token_ids"),
            prompt_patch_count,
        )
        return finalized_prompt


def build_ming_ingress_processor(*, vllm_config: Any, tokenizer: Any) -> MingIngressProcessor:
    return MingIngressProcessor(vllm_config=vllm_config, tokenizer=tokenizer)
