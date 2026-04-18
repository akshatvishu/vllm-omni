# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    AUDIO_FRAME_HOP,
    KEY_CFG,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_PROMPT_LATENTS,
    KEY_SPEAKER_EMBEDDING,
    PATCH_SIZE,
    SAMPLE_RATE,
)
from vllm_omni.model_executor.models.ming_tts.ingress import MingIngressProcessor
from vllm_omni.model_executor.models.ming_tts.prompt_builder import (
    build_dense_prompt_token_ids,
    build_ming_dense_prompt,
    count_prompt_waveform_patches,
    pad_prompt_waveform,
)


class _DummyTokenizer:
    def __init__(self):
        self._token_to_id = {"<audioPatch>": 9001, "<|vision_start|>": 9002}
        self._id_to_token = {token_id: token for token, token_id in self._token_to_id.items()}
        self._next = 100

    def encode(self, text):
        if text not in self._token_to_id:
            self._token_to_id[text] = self._next
            self._id_to_token[self._next] = text
            self._next += 1
        return [self._token_to_id[text]]

    def convert_tokens_to_ids(self, token):
        if token not in self._token_to_id:
            self._token_to_id[token] = self._next
            self._id_to_token[self._next] = token
            self._next += 1
        return self._token_to_id[token]

    def decode(self, token_ids):
        return "".join(self._id_to_token[int(token_id)] for token_id in token_ids)


def _make_dummy_ingress_processor(tokenizer):
    processor = MingIngressProcessor.__new__(MingIngressProcessor)
    processor.tokenizer = tokenizer
    processor.profile_ingress = False
    processor.ming_config = SimpleNamespace(patch_size=4, latent_dim=64, vae_patch_size=4, audio_frame_hop=882)
    return processor


def test_build_dense_prompt_token_ids_matches_ming_dense_layout():
    tokenizer = _DummyTokenizer()

    prompt_ids = build_dense_prompt_token_ids(
        tokenizer,
        prompt="Prompt text.",
        text="Target text.",
        instruction="instruction-json",
        prompt_text="reference transcript",
        speaker_count=2,
        prompt_patch_count=3,
    )

    assert prompt_ids.count(tokenizer.convert_tokens_to_ids("<audioPatch>")) == 3
    assert prompt_ids.count(tokenizer.convert_tokens_to_ids("<|vision_start|>")) == 2
    assert tokenizer.encode("instruction-json")[0] in prompt_ids
    assert tokenizer.encode("reference transcript")[0] in prompt_ids


def test_build_ming_dense_prompt_pads_prompt_waveform_and_zero_speaker():
    tokenizer = _DummyTokenizer()
    waveform = torch.ones((1, 1000), dtype=torch.float32)

    prompt = build_ming_dense_prompt(
        tokenizer,
        prompt="Please imitate the reference speech.",
        text="Hello world.",
        prompt_text="Reference words.",
        prompt_waveform=waveform,
        use_zero_spk_emb=True,
    )

    info = prompt["additional_information"]
    padded_waveform = info["prompt_waveform"]

    assert padded_waveform.shape == (1, 14112)
    assert int(info[KEY_SPEAKER_EMBEDDING].numel()) == 192
    expected_patch_count = count_prompt_waveform_patches(waveform)
    assert prompt["prompt_token_ids"].count(tokenizer.convert_tokens_to_ids("<audioPatch>")) == expected_patch_count


def test_build_ming_dense_prompt_uses_patch_count_not_frame_count_for_zero_shot_waveform():
    tokenizer = _DummyTokenizer()
    waveform = torch.ones((1, 211680), dtype=torch.float32)

    prompt = build_ming_dense_prompt(
        tokenizer,
        prompt="Please generate speech based on the following description.\n",
        text="Target text.",
        prompt_text="Reference words.",
        prompt_waveform=waveform,
        speaker_embedding=torch.ones((192,), dtype=torch.float32),
    )

    expected_patch_count = count_prompt_waveform_patches(waveform)
    assert prompt["additional_information"].get(KEY_PROMPT_LATENTS) is None
    assert prompt["prompt_token_ids"].count(tokenizer.convert_tokens_to_ids("<audioPatch>")) == expected_patch_count


def test_build_ming_dense_prompt_accepts_flat_speaker_embedding_list():
    tokenizer = _DummyTokenizer()
    speaker_embedding = [0.1] * 192

    prompt = build_ming_dense_prompt(
        tokenizer,
        prompt="Please imitate the reference speech.",
        text="Hello world.",
        speaker_embedding=speaker_embedding,
    )

    info = prompt["additional_information"]
    assert tuple(info[KEY_SPEAKER_EMBEDDING].shape) == (192,)
    assert prompt["prompt_token_ids"].count(tokenizer.convert_tokens_to_ids("<|vision_start|>")) == 1


def test_build_ming_dense_prompt_uses_prompt_latents_to_set_patch_count():
    tokenizer = _DummyTokenizer()
    prompt_latents = torch.ones((15, 4, 64), dtype=torch.float32)

    prompt = build_ming_dense_prompt(
        tokenizer,
        prompt="Please generate speech based on the following description.\n",
        text="Target text.",
        prompt_text="Reference words.",
        prompt_latents=prompt_latents,
        speaker_embedding=torch.ones((192,), dtype=torch.float32),
    )

    assert torch.equal(prompt["additional_information"][KEY_PROMPT_LATENTS], prompt_latents)
    assert prompt["prompt_token_ids"].count(tokenizer.convert_tokens_to_ids("<audioPatch>")) == 15


def test_build_ming_dense_prompt_allows_raw_waveform_shell_without_explicit_prompt_latents():
    tokenizer = _DummyTokenizer()
    waveform = torch.ones((1, 1000), dtype=torch.float32)

    prompt = build_ming_dense_prompt(
        tokenizer,
        prompt="Please imitate the reference speech.",
        text="Hello world.",
        prompt_text="Reference words.",
        prompt_waveform=waveform,
        speaker_embedding=torch.ones((192,), dtype=torch.float32),
    )

    expected_patch_count = count_prompt_waveform_patches(waveform)
    assert prompt["additional_information"].get(KEY_PROMPT_LATENTS) is None
    assert prompt["prompt_token_ids"].count(tokenizer.convert_tokens_to_ids("<audioPatch>")) == expected_patch_count


def test_build_ming_dense_prompt_rejects_dual_truth_waveform_and_prompt_latents():
    tokenizer = _DummyTokenizer()
    waveform = torch.ones((1, 1000), dtype=torch.float32)
    prompt_latents = torch.ones((4, 64), dtype=torch.float32)

    with pytest.raises(ValueError, match="Choose exactly one source of truth"):
        build_ming_dense_prompt(
            tokenizer,
            prompt="Please imitate the reference speech.",
            text="Hello world.",
            prompt_text="Reference words.",
            prompt_waveform=waveform,
            prompt_latents=prompt_latents,
        )


def test_ming_ingress_processor_preserves_raw_waveform_for_stage0_encoding():
    tokenizer = _DummyTokenizer()
    waveform = torch.ones((1, 1000), dtype=torch.float32)
    prompt_text = "Reference words."
    prompt = build_ming_dense_prompt(
        tokenizer,
        prompt="Please imitate the reference speech.",
        text="Hello world.",
        prompt_text=prompt_text,
        prompt_waveform=waveform,
        speaker_embedding=torch.ones((192,), dtype=torch.float32),
    )
    prompt["prompt"] = "Please imitate the reference speech."
    prompt["text"] = "Hello world."
    prompt["prompt_text"] = prompt_text
    prompt["prompt_waveform"] = waveform
    prompt["prompt_waveform_length"] = torch.tensor([1000], dtype=torch.int32)

    processor = _make_dummy_ingress_processor(tokenizer)
    finalized = processor(prompt)

    assert finalized["prompt_waveform"] is waveform
    assert torch.equal(finalized["prompt_waveform_length"], torch.tensor([1000], dtype=torch.int32))
    assert finalized["additional_information"]["prompt_waveform"] is prompt["additional_information"]["prompt_waveform"]
    assert torch.equal(
        finalized["additional_information"]["prompt_waveform_length"],
        prompt["additional_information"]["prompt_waveform_length"],
    )
    assert KEY_PROMPT_LATENTS not in finalized["additional_information"]
    expected_patch_count = count_prompt_waveform_patches(waveform)
    assert finalized["prompt_token_ids"].count(tokenizer.convert_tokens_to_ids("<audioPatch>")) == expected_patch_count


def test_build_ming_dense_prompt_rejects_prompt_waveform_without_prompt_text():
    tokenizer = _DummyTokenizer()
    waveform = torch.ones((1, 1000), dtype=torch.float32)

    with pytest.raises(ValueError, match="prompt_waveform requires prompt_text"):
        build_ming_dense_prompt(
            tokenizer,
            prompt="Please generate speech based on the following description.\n",
            text="我竟然抢到了陈奕迅的演唱会门票！",
            instruction={"情感": "高兴"},
            prompt_waveform=waveform,
        )


def test_ming_ingress_processor_rejects_raw_prompt_waveform_without_prompt_text():
    tokenizer = _DummyTokenizer()
    waveform = torch.ones((1, 1000), dtype=torch.float32)
    prompt = {
        "prompt": "Please generate speech based on the following description.\n",
        "text": "我竟然抢到了陈奕迅的演唱会门票！",
        "prompt_token_ids": [1, 2, 3],
        "additional_information": {
            "prompt_waveform": waveform,
            "prompt_waveform_length": torch.tensor([1000], dtype=torch.int32),
        },
    }

    processor = _make_dummy_ingress_processor(tokenizer)

    with pytest.raises(RuntimeError, match="prompt_waveform requires prompt_text"):
        processor(prompt)


def test_ming_ingress_processor_rebuilds_podcast_prompt_with_prompt_text_before_target_text():
    tokenizer = _DummyTokenizer()
    prompt_prefix = "Please generate speech based on the following description.\n"
    prompt_text = " speaker_1:reference one\n speaker_2:reference two\n"
    target_text = " speaker_1:target one\n speaker_2:target two\n"
    speaker_embeddings = torch.ones((2, 192), dtype=torch.float32)
    prompt_waveform = [
        torch.ones((1, 1000), dtype=torch.float32),
        torch.ones((1, 2000), dtype=torch.float32),
    ]

    prompt = build_ming_dense_prompt(
        tokenizer,
        prompt=prompt_prefix,
        text=target_text,
        prompt_text=prompt_text,
        prompt_waveform=prompt_waveform,
        speaker_embedding=speaker_embeddings,
    )

    processor = _make_dummy_ingress_processor(tokenizer)
    finalized = processor(prompt)
    decoded = tokenizer.decode(finalized["prompt_token_ids"])
    expected_patch_count = count_prompt_waveform_patches(prompt_waveform)

    assert decoded.index(prompt_text) < decoded.index(target_text)
    assert finalized["prompt_token_ids"].count(tokenizer.convert_tokens_to_ids("<|vision_start|>")) == 2
    assert finalized["prompt_token_ids"].count(tokenizer.convert_tokens_to_ids("<audioPatch>")) == expected_patch_count
    assert "prompt_waveform" in finalized["additional_information"]
    assert KEY_PROMPT_LATENTS not in finalized["additional_information"]


def test_build_ming_dense_prompt_keeps_single_speaker_initial_payload_compatible():
    tokenizer = _DummyTokenizer()
    prompt_prefix = "Please imitate the reference speech."
    target_text = "Hello world."
    prompt_text = "Reference words."
    waveform = torch.ones((1, 1000), dtype=torch.float32)

    prompt = build_ming_dense_prompt(
        tokenizer,
        prompt=prompt_prefix,
        text=target_text,
        prompt_text=prompt_text,
        prompt_waveform=waveform,
        speaker_embedding=torch.ones((192,), dtype=torch.float32),
    )
    expected_patch_count = count_prompt_waveform_patches(waveform)
    expected_prompt_token_ids = build_dense_prompt_token_ids(
        tokenizer,
        prompt=prompt_prefix,
        text=target_text,
        prompt_text=prompt_text,
        speaker_count=1,
        prompt_patch_count=expected_patch_count,
    )

    assert prompt["prompt"] == prompt_prefix
    assert prompt["text"] == target_text
    assert prompt["prompt_token_ids"] == expected_prompt_token_ids
    assert prompt["prompt_token_ids"].count(tokenizer.convert_tokens_to_ids("<audioPatch>")) == expected_patch_count
    assert prompt["additional_information"]["prompt_text"] == prompt_text


def test_pad_prompt_waveform_matches_upstream_ming_alignment():
    padded = pad_prompt_waveform(torch.ones((1, 3529), dtype=torch.float32))
    assert int(padded.shape[-1]) == 14112
    assert int(padded.shape[-1]) % int((float(SAMPLE_RATE) / 12.5) * int(PATCH_SIZE)) == 0
    assert int(padded.shape[-1]) % int(AUDIO_FRAME_HOP * PATCH_SIZE) == 0


def test_build_ming_dense_prompt_injects_duration_window_when_missing():
    tokenizer = _DummyTokenizer()

    prompt = build_ming_dense_prompt(
        tokenizer,
        prompt="Please generate music based on the following description.\n",
        text=" Genre: electronic. Mood: confident. Instrument: drums. Theme: festival. Duration: 30s.",
        runtime_controls={KEY_CFG: 2.0},
    )

    info = prompt["additional_information"]
    assert float(info[KEY_CFG].item()) == 2.0
    assert int(info[KEY_MIN_DECODE_STEPS].item()) == 91
    assert int(info[KEY_MAX_DECODE_STEPS].item()) == 97


def test_build_ming_dense_prompt_preserves_explicit_decode_window_overrides():
    tokenizer = _DummyTokenizer()

    prompt = build_ming_dense_prompt(
        tokenizer,
        prompt="Please generate music based on the following description.\n",
        text=" Genre: electronic. Mood: confident. Instrument: drums. Theme: festival. Duration: 30s.",
        runtime_controls={
            KEY_MIN_DECODE_STEPS: 11,
            KEY_MAX_DECODE_STEPS: 13,
        },
    )

    info = prompt["additional_information"]
    assert int(info[KEY_MIN_DECODE_STEPS].item()) == 11
    assert int(info[KEY_MAX_DECODE_STEPS].item()) == 13


def test_build_ming_dense_prompt_does_not_inject_duration_window_without_valid_duration():
    tokenizer = _DummyTokenizer()

    prompt_missing = build_ming_dense_prompt(
        tokenizer,
        prompt="Please generate music based on the following description.\n",
        text=" Genre: electronic. Mood: confident. Instrument: drums. Theme: festival.",
        runtime_controls={KEY_CFG: 2.0},
    )
    prompt_malformed = build_ming_dense_prompt(
        tokenizer,
        prompt="Please generate music based on the following description.\n",
        text=" Genre: electronic. Mood: confident. Instrument: drums. Theme: festival. Duration: nope.",
        runtime_controls={KEY_CFG: 2.0},
    )

    for prompt in (prompt_missing, prompt_malformed):
        info = prompt["additional_information"]
        assert KEY_MIN_DECODE_STEPS not in info
        assert KEY_MAX_DECODE_STEPS not in info
