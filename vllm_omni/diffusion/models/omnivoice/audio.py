# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audio preparation and output processing for OmniVoice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, detect_nonsilent, split_on_silence

_END_PUNCTUATION = {
    ";",
    ":",
    ",",
    ".",
    "!",
    "?",
    "…",
    ")",
    "]",
    "}",
    '"',
    "'",
    "“",
    "”",
    "‘",
    "’",
    "；",
    "：",
    "，",
    "。",
    "！",
    "？",
    "、",
    "……",
    "）",
    "】",
}


@dataclass(frozen=True)
class PreparedReferenceAudio:
    """Reference waveform shared by ASR and the audio tokenizer."""

    waveform: np.ndarray
    sample_rate: int
    original_rms: float


def _numpy_to_audio_segment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert a channel-first float waveform to an in-memory PCM segment."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    audio_int = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    if audio_int.shape[0] > 1:
        audio_int = audio_int.T.reshape(-1)
    else:
        audio_int = audio_int.reshape(-1)
    return AudioSegment(
        data=audio_int.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=audio.shape[0],
    )


def _audio_segment_to_numpy(audio: AudioSegment) -> np.ndarray:
    """Convert an in-memory PCM segment to a channel-first float waveform."""
    data = np.asarray(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    if audio.channels == 1:
        return data[np.newaxis, :]
    return data.reshape(-1, audio.channels).T


def remove_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    middle_silence_ms: int,
    leading_silence_ms: int,
    trailing_silence_ms: int,
) -> np.ndarray:
    """Remove long gaps and trim edge silence using official thresholds."""
    wave = _numpy_to_audio_segment(audio, sample_rate)
    if middle_silence_ms > 0:
        non_silent_segments = split_on_silence(
            wave,
            min_silence_len=middle_silence_ms,
            silence_thresh=-50,
            keep_silence=middle_silence_ms,
            seek_step=10,
        )
        wave = AudioSegment.silent(duration=0)
        for segment in non_silent_segments:
            wave += segment

    start_idx = detect_leading_silence(wave, silence_threshold=-50)
    start_idx = max(0, start_idx - leading_silence_ms)
    wave = wave[start_idx:]

    wave = wave.reverse()
    start_idx = detect_leading_silence(wave, silence_threshold=-50)
    start_idx = max(0, start_idx - trailing_silence_ms)
    wave = wave[start_idx:].reverse()

    return _audio_segment_to_numpy(wave)


def trim_long_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_duration_s: float = 15.0,
    min_duration_s: float = 3.0,
    trim_threshold_s: float = 20.0,
) -> np.ndarray:
    """Trim long reference audio at a suitable silence boundary."""
    if audio.shape[-1] / sample_rate <= trim_threshold_s:
        return audio

    segment = _numpy_to_audio_segment(audio, sample_rate)
    non_silent_ranges = detect_nonsilent(
        segment,
        min_silence_len=100,
        silence_thresh=-40,
        seek_step=10,
    )
    if not non_silent_ranges:
        return audio

    max_ms = int(max_duration_s * 1000)
    min_ms = int(min_duration_s * 1000)
    best_split = 0
    for start, end in non_silent_ranges:
        if start > best_split and start <= max_ms:
            best_split = start
        if end > max_ms:
            break

    if best_split < min_ms:
        best_split = min(max_ms, len(segment))
    return _audio_segment_to_numpy(segment[:best_split])


def prepare_reference_audio(
    waveform: np.ndarray | torch.Tensor,
    sample_rate: int,
    *,
    target_sample_rate: int,
    hop_length: int,
    trim_long: bool,
) -> PreparedReferenceAudio:
    """Prepare reference audio before ASR and audio-tokenizer encoding."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.array(waveform, dtype=np.float32, copy=True)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    elif waveform.ndim != 2:
        raise ValueError(f"OmniVoice reference audio must be 1D or 2D, got {waveform.ndim} dimensions.")
    if not waveform.size or not np.any(waveform):
        raise ValueError("Reference audio is empty after silence removal.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0, keepdims=True)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            torch.from_numpy(waveform),
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        ).numpy()

    original_rms = float(np.sqrt(np.mean(waveform**2))) if waveform.size else 0.0
    if 0 < original_rms < 0.1:
        waveform = waveform * (0.1 / original_rms)

    if trim_long:
        waveform = trim_long_audio(waveform, target_sample_rate)
    waveform = remove_silence(
        waveform,
        target_sample_rate,
        middle_silence_ms=200,
        leading_silence_ms=100,
        trailing_silence_ms=200,
    )
    if waveform.shape[-1] == 0:
        raise ValueError("Reference audio is empty after silence removal.")

    remainder = waveform.shape[-1] % hop_length
    if remainder:
        waveform = waveform[:, :-remainder]
    if waveform.shape[-1] == 0:
        raise ValueError("Reference audio is shorter than one audio-tokenizer hop after preprocessing.")

    return PreparedReferenceAudio(
        waveform=np.ascontiguousarray(waveform, dtype=np.float32),
        sample_rate=target_sample_rate,
        original_rms=original_rms,
    )


def add_reference_punctuation(text: str) -> str:
    """Add an English or Chinese sentence terminator when one is missing."""
    text = text.strip()
    if not text:
        return text
    if text[-1] in _END_PUNCTUATION:
        return text
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return f"{text}。"
    return f"{text}."


def postprocess_generated_audio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    reference_rms: float | None,
) -> np.ndarray:
    """Apply official OmniVoice output silence, volume, fade, and padding rules."""
    audio = np.array(audio, dtype=np.float32, copy=True)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.ndim != 2:
        raise ValueError(f"OmniVoice generated audio must be 1D or 2D, got {audio.ndim} dimensions.")

    audio = remove_silence(
        audio,
        sample_rate,
        middle_silence_ms=500,
        leading_silence_ms=100,
        trailing_silence_ms=100,
    )

    if audio.shape[-1] == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)

    if reference_rms is not None and reference_rms < 0.1:
        audio = audio * (reference_rms / 0.1)
    elif reference_rms is None:
        peak = float(np.abs(audio).max())
        if peak > 1e-6:
            audio = audio / peak * 0.5

    fade_samples = int(0.1 * sample_rate)
    if fade_samples > 0:
        fade_length = min(fade_samples, audio.shape[-1] // 2)
        if fade_length > 0:
            fade_in = np.linspace(0, 1, fade_length, dtype=np.float32)[np.newaxis, :]
            fade_out = np.linspace(1, 0, fade_length, dtype=np.float32)[np.newaxis, :]
            audio[:, :fade_length] *= fade_in
            audio[:, -fade_length:] *= fade_out

    pad_samples = int(0.1 * sample_rate)
    if pad_samples > 0:
        padding = np.zeros((audio.shape[0], pad_samples), dtype=np.float32)
        audio = np.concatenate([padding, audio, padding], axis=-1)
    return np.ascontiguousarray(audio, dtype=np.float32)
