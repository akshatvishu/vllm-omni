# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig, Qwen2Config

from .audio_tokenizer.configuration_audio_vae import AudioVAEconfig


def _coerce_qwen2_config(value: Any) -> Qwen2Config:
    if isinstance(value, Qwen2Config):
        return value
    if isinstance(value, PretrainedConfig):
        return Qwen2Config.from_dict(value.to_dict())
    if isinstance(value, dict):
        return Qwen2Config.from_dict(dict(value))
    raise TypeError(f"Unsupported llm_config type for Ming dense config: {type(value)!r}")


def _coerce_audio_vae_config(value: Any) -> AudioVAEconfig | None:
    if value is None:
        return None
    if isinstance(value, AudioVAEconfig):
        value = value.to_dict()
    elif isinstance(value, PretrainedConfig):
        value = value.to_dict()
    elif isinstance(value, dict):
        value = dict(value)
    else:
        raise TypeError(f"Unsupported audio_tokenizer_config type for Ming dense config: {type(value)!r}")

    return AudioVAEconfig(**value)


class MingDenseConfig(PretrainedConfig):
    model_type = "dense"

    def __init__(
        self,
        llm_config: Qwen2Config | dict[str, Any] | None = None,
        ditar_config: dict[str, Any] | None = None,
        aggregator_config: dict[str, Any] | None = None,
        audio_tokenizer_config: AudioVAEconfig | dict[str, Any] | None = None,
        architectures: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(architectures=architectures, **kwargs)
        self.llm_config = _coerce_qwen2_config(llm_config or {})
        self.ditar_config = dict(ditar_config or {})
        self.aggregator_config = dict(aggregator_config or {})
        self.audio_tokenizer_config = _coerce_audio_vae_config(audio_tokenizer_config)

    def get_text_config(self, decoder: bool = False, **kwargs: Any) -> Qwen2Config:
        del decoder, kwargs
        return self.llm_config
