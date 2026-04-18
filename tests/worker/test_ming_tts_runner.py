# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_LATENT_HISTORY,
    KEY_NEXT_EMBEDS,
    KEY_PROMPT_LATENTS,
    KEY_REQUEST_ID,
    KEY_SPEAKER_EMBEDDING,
    MingTTSConfig,
)
from vllm_omni.model_executor.models.ming_tts.ming_tts import MingTTSForConditionalGeneration
from vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae import MingAudioVAEModel
from vllm_omni.model_executor.models.ming_tts.ming_tts_llm import (
    MING_STOP_REASON_CONTINUE,
    MING_STOP_REASON_KEY,
    MING_STOP_REASON_MAX_DECODE_STEPS,
    MING_STOP_REASON_STOP_HEAD,
    MingLLMModel,
    _resolve_ming_stop_decision,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DummyBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.to(torch.float32).reshape(-1, 1)
        return ids.repeat(1, self.hidden_size) / 100.0

    def get_input_embeddings(self):
        return None

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        del input_ids, positions, intermediate_tensors, kwargs
        return inputs_embeds


class NaNOnSecondDecodeBackbone(DummyBackbone):
    def __init__(self, hidden_size: int):
        super().__init__(hidden_size)
        self.decode_calls = 0

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        del input_ids, positions, intermediate_tensors, kwargs
        self.decode_calls += 1
        if self.decode_calls >= 2:
            return torch.full_like(inputs_embeds, float("nan"))
        return inputs_embeds


class DummyAggregator(torch.nn.Module):
    def __init__(self, in_channels: int, llm_input_dim: int, **kwargs):
        super().__init__()
        del in_channels, kwargs
        self.hidden_size = llm_input_dim

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        pooled = patch.mean(dim=1)
        repeats = self.hidden_size // pooled.shape[-1]
        return pooled.repeat(1, repeats).reshape(pooled.shape[0], 1, self.hidden_size)


class DummyFlowLoss(torch.nn.Module):
    def __init__(self, z_channels: int, llm_cond_dim: int, **kwargs):
        super().__init__()
        del z_channels, llm_cond_dim, kwargs

    def sample(self, z, latent_history, cfg, patch_size, sigma, temperature):
        del latent_history, cfg, sigma, temperature
        base = z[:, 0, :64]
        return torch.stack([base + float(i + 1) for i in range(patch_size)], dim=1)


class DummyAudioVAE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.decode_calls: list[dict[str, object]] = []

    def encode_latent(self, waveform: torch.Tensor, waveform_length: torch.Tensor):
        if waveform.ndim == 2:
            frames = waveform.shape[-1] // 64
            latent = waveform[:, : frames * 64].reshape(waveform.shape[0], frames, 64)
        else:
            latent = waveform
        frame_num = torch.full((latent.shape[0],), latent.shape[1], dtype=torch.int32, device=latent.device)
        return latent.to(torch.float32), frame_num

    def decode(self, latent, past_key_values=None, use_cache=False, stream_state=(None, None, None), last_chunk=False):
        del use_cache, last_chunk
        prev_frames = int((past_key_values or {}).get("frames", 0))
        waveform = latent.sum(dim=-1).reshape(latent.shape[0], -1).to(torch.float32) + prev_frames * 10.0
        new_stream_state = ("stream", prev_frames + latent.shape[1], tuple(latent.shape))
        new_past = {"frames": prev_frames + int(latent.shape[1])}
        self.decode_calls.append(
            {
                "stream_state": stream_state,
                "past_key_values": past_key_values,
                "latent_shape": tuple(latent.shape),
            }
        )
        return waveform, new_stream_state, new_past


class _DummySamplingMetadata:
    def __init__(self, step: int):
        self.output_token_ids = [[0] * int(step)]


def _make_config() -> MingTTSConfig:
    audio_cfg = SimpleNamespace(
        enc_kwargs={"latent_dim": 64, "input_dim": 882, "hop_size": 882},
        dec_kwargs={"latent_dim": 64, "output_dim": 882},
        patch_size=4,
        sample_rate=44100,
    )
    cfg = MingTTSConfig(audio_tokenizer_config=audio_cfg)
    cfg.validate()
    return cfg


def _make_vllm_config(model_stage: str, **hf_overrides):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(**hf_overrides), model_stage=model_stage),
        quant_config=None,
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )


def _make_runner_for_ming(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as wrapper_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae as vae_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))

    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: DummyBackbone(cfg.llm_hidden_size))
    monkeypatch.setattr(llm_mod, "Aggregator", DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", DummyFlowLoss)
    monkeypatch.setattr(wrapper_mod, "AudioVAE", DummyAudioVAE, raising=False)
    monkeypatch.setattr(vae_mod, "AudioVAE", DummyAudioVAE)

    llm_model = MingLLMModel(vllm_config=_make_vllm_config("llm"))
    vae_model = MingAudioVAEModel(vllm_config=_make_vllm_config("audio_vae"))

    def _wrapper_loader(*, architectures, **kwargs):
        arch = architectures[0]
        if arch == "MingLLMModel":
            return llm_model
        if arch == "MingAudioVAEModel":
            return vae_model
        raise AssertionError(f"unexpected architecture {arch}")

    monkeypatch.setattr(wrapper_mod, "init_vllm_registered_model", _wrapper_loader)

    stage1 = MingTTSForConditionalGeneration(vllm_config=_make_vllm_config("llm"))
    stage2 = MingTTSForConditionalGeneration(vllm_config=_make_vllm_config("audio_vae"))

    return SimpleNamespace(config=cfg, llm=llm_model, vae=vae_model, stage1=stage1, stage2=stage2)


def test_ming_llm_step_shapes(monkeypatch):
    runner = _make_runner_for_ming(monkeypatch)
    cfg = runner.config

    prefill_ids = torch.tensor(
        [1, cfg.audio_start_token_id, cfg.audio_dummy_token_id, cfg.audio_dummy_token_id, cfg.audio_end_token_id, 2],
        dtype=torch.long,
    )
    prefill_embeds = torch.zeros((prefill_ids.shape[0], cfg.llm_hidden_size), dtype=torch.float32)
    prompt_latents = torch.arange(8 * 64, dtype=torch.float32).reshape(1, 8, 64)

    _, prefill_out_embeds, prefill_info = runner.stage1.preprocess_input(
        prefill_ids,
        prefill_embeds,
        **{KEY_PROMPT_LATENTS: prompt_latents},
        **{KEY_REQUEST_ID: "req-1"},
    )

    assert prefill_info[KEY_LATENT_HISTORY].shape == (32, 64)
    assert torch.allclose(prefill_info[KEY_LATENT_HISTORY][-8:], prompt_latents.reshape(8, 64))
    assert torch.count_nonzero(prefill_out_embeds[1]).item() > 0
    assert torch.count_nonzero(prefill_out_embeds[2]).item() > 0

    decode_ids = torch.tensor([cfg.audio_dummy_token_id], dtype=torch.long)
    decode_embeds = torch.zeros((1, cfg.llm_hidden_size), dtype=torch.float32)
    _, decode_embeds, decode_info = runner.stage1.preprocess_input(
        decode_ids,
        decode_embeds,
        **prefill_info,
    )

    output = runner.llm.forward(
        decode_ids,
        positions=torch.tensor([0], dtype=torch.long),
        inputs_embeds=decode_embeds,
        model_intermediate_buffer=[decode_info],
        seq_token_counts=[1],
    )
    mm = output.multimodal_outputs

    assert mm["ming_latent_patch"].shape == (1, 4, 64)
    assert mm["ming_next_embeds"].shape == (1, 1, cfg.llm_hidden_size)
    assert mm["ming_new_history"].shape == (1, 32, 64)

    update = runner.stage1.postprocess(output.text_hidden_states, multimodal_outputs=mm, **decode_info)
    assert update[KEY_LATENT_HISTORY].shape == (1, 32, 64)
    assert torch.allclose(update[KEY_LATENT_HISTORY][0, -4:], mm["ming_latent_patch"][0].cpu())
    assert update[KEY_NEXT_EMBEDS].shape == (1, 1, cfg.llm_hidden_size)


def test_ming_prefill_injects_speaker_into_dense_placeholder(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as wrapper_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: DummyBackbone(cfg.llm_hidden_size))
    monkeypatch.setattr(llm_mod, "Aggregator", DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", DummyFlowLoss)
    monkeypatch.setattr(
        wrapper_mod, "init_vllm_registered_model", lambda **kwargs: MingLLMModel(vllm_config=_make_vllm_config("llm"))
    )

    vision_start_token_id = 32001
    stage1 = MingTTSForConditionalGeneration(
        vllm_config=_make_vllm_config("llm", vision_start_token_id=vision_start_token_id)
    )

    input_ids = torch.tensor(
        [
            1,
            vision_start_token_id,
            77,
            cfg.audio_start_token_id,
            cfg.audio_dummy_token_id,
            cfg.audio_end_token_id,
        ],
        dtype=torch.long,
    )
    input_embeds = torch.zeros((input_ids.shape[0], cfg.llm_hidden_size), dtype=torch.float32)
    baseline_embeds = stage1.model.embed_input_ids(input_ids).clone()
    speaker = torch.ones((192,), dtype=torch.float32)

    _, out_embeds, _ = stage1.preprocess_input(
        input_ids,
        input_embeds,
        **{KEY_SPEAKER_EMBEDDING: speaker},
    )

    assert torch.count_nonzero(out_embeds[2]).item() > 0
    assert not torch.allclose(out_embeds[2], baseline_embeds[2])
    assert torch.allclose(out_embeds[3], baseline_embeds[3])


def test_ming_prefill_injects_multiple_speakers_into_multiple_dense_placeholders(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as wrapper_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: DummyBackbone(cfg.llm_hidden_size))
    monkeypatch.setattr(llm_mod, "Aggregator", DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", DummyFlowLoss)
    monkeypatch.setattr(
        wrapper_mod, "init_vllm_registered_model", lambda **kwargs: MingLLMModel(vllm_config=_make_vllm_config("llm"))
    )

    vision_start_token_id = 32001
    stage1 = MingTTSForConditionalGeneration(
        vllm_config=_make_vllm_config("llm", vision_start_token_id=vision_start_token_id)
    )

    input_ids = torch.tensor(
        [
            1,
            vision_start_token_id,
            77,
            2,
            vision_start_token_id,
            88,
            cfg.audio_start_token_id,
            cfg.audio_dummy_token_id,
            cfg.audio_end_token_id,
        ],
        dtype=torch.long,
    )
    input_embeds = torch.zeros((input_ids.shape[0], cfg.llm_hidden_size), dtype=torch.float32)
    baseline_embeds = stage1.model.embed_input_ids(input_ids).clone()
    speaker = torch.ones((2, 192), dtype=torch.float32)

    _, out_embeds, _ = stage1.preprocess_input(
        input_ids,
        input_embeds,
        **{KEY_SPEAKER_EMBEDDING: speaker},
    )

    assert torch.count_nonzero(out_embeds[2]).item() > 0
    assert torch.count_nonzero(out_embeds[5]).item() > 0
    assert not torch.allclose(out_embeds[2], baseline_embeds[2])
    assert not torch.allclose(out_embeds[5], baseline_embeds[5])
    assert torch.allclose(out_embeds[6], baseline_embeds[6])


def test_ming_stop_logic_no_stop_before_min_required_decode_steps(monkeypatch):
    runner = _make_runner_for_ming(monkeypatch)
    cfg = runner.config

    def _high_stop(_hidden_states):
        return torch.tensor([[0.0, 10.0]], dtype=torch.float32)

    monkeypatch.setattr(runner.llm.stop_head, "forward", _high_stop)
    hidden = torch.zeros((1, cfg.llm_hidden_size), dtype=torch.float32)

    stop_reason, stop_now, force_stop, min_required_decode_steps, next_token_id = _resolve_ming_stop_decision(
        step=4,
        stop_prob=1.0,
        stop_threshold=float(cfg.stop_head_threshold),
        min_stop_step=int(cfg.stop_head_min_steps),
        min_decode_steps=7,
        max_decode_steps=int(cfg.max_decode_steps),
        audio_dummy_token_id=int(cfg.audio_dummy_token_id),
        text_eos_token_id=int(cfg.text_eos_token_id),
    )
    assert stop_reason == MING_STOP_REASON_CONTINUE
    assert stop_now is False
    assert force_stop is False
    assert min_required_decode_steps == 7
    assert next_token_id == cfg.audio_dummy_token_id

    logits_step3 = runner.llm.compute_logits(
        OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"ming_min_decode_steps": torch.tensor([7], dtype=torch.int32)},
        ),
        _DummySamplingMetadata(step=3),
    )
    out_step3 = runner.llm.sample(logits_step3, _DummySamplingMetadata(step=3))
    assert int(out_step3.sampled_token_ids[0, 0]) == cfg.audio_dummy_token_id
    assert torch.isfinite(logits_step3[0, int(cfg.audio_dummy_token_id)])
    assert not torch.isfinite(logits_step3[0, int(cfg.text_eos_token_id)])


def test_ming_stop_logic_stop_head_inside_window(monkeypatch):
    runner = _make_runner_for_ming(monkeypatch)
    cfg = runner.config

    def _high_stop(_hidden_states):
        return torch.tensor([[0.0, 10.0]], dtype=torch.float32)

    monkeypatch.setattr(runner.llm.stop_head, "forward", _high_stop)
    hidden = torch.zeros((1, cfg.llm_hidden_size), dtype=torch.float32)

    stop_reason, stop_now, force_stop, min_required_decode_steps, next_token_id = _resolve_ming_stop_decision(
        step=4,
        stop_prob=1.0,
        stop_threshold=float(cfg.stop_head_threshold),
        min_stop_step=int(cfg.stop_head_min_steps),
        min_decode_steps=0,
        max_decode_steps=int(cfg.max_decode_steps),
        audio_dummy_token_id=int(cfg.audio_dummy_token_id),
        text_eos_token_id=int(cfg.text_eos_token_id),
    )
    assert stop_reason == MING_STOP_REASON_STOP_HEAD
    assert stop_now is True
    assert force_stop is False
    assert min_required_decode_steps == int(cfg.stop_head_min_steps) + 1
    assert next_token_id == cfg.text_eos_token_id

    logits_step4 = runner.llm.compute_logits(hidden, _DummySamplingMetadata(step=4))
    out_step4 = runner.llm.sample(logits_step4, _DummySamplingMetadata(step=4))
    assert int(out_step4.sampled_token_ids[0, 0]) == cfg.text_eos_token_id


def test_ming_stop_logic_rejects_impossible_decode_window(monkeypatch):
    runner = _make_runner_for_ming(monkeypatch)
    cfg = runner.config
    hidden = torch.zeros((1, cfg.llm_hidden_size), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Invalid Ming decode window"):
        runner.llm.compute_logits(
            OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={
                    "ming_min_decode_steps": torch.tensor([7], dtype=torch.int32),
                    "ming_max_decode_steps": torch.tensor([5], dtype=torch.int32),
                },
            ),
            _DummySamplingMetadata(step=4),
        )


def test_ming_stop_logic_max_decode_guard(monkeypatch):
    runner = _make_runner_for_ming(monkeypatch)
    cfg = runner.config
    cfg.max_decode_steps = 5

    def _high_stop(_hidden_states):
        return torch.tensor([[0.0, 10.0]], dtype=torch.float32)

    monkeypatch.setattr(runner.llm.stop_head, "forward", _high_stop)
    hidden = torch.zeros((1, cfg.llm_hidden_size), dtype=torch.float32)

    stop_reason, stop_now, force_stop, min_required_decode_steps, next_token_id = _resolve_ming_stop_decision(
        step=4,
        stop_prob=1.0,
        stop_threshold=float(cfg.stop_head_threshold),
        min_stop_step=int(cfg.stop_head_min_steps),
        min_decode_steps=0,
        max_decode_steps=int(cfg.max_decode_steps),
        audio_dummy_token_id=int(cfg.audio_dummy_token_id),
        text_eos_token_id=int(cfg.text_eos_token_id),
    )
    assert stop_reason == MING_STOP_REASON_MAX_DECODE_STEPS
    assert stop_now is True
    assert force_stop is True
    assert min_required_decode_steps == int(cfg.stop_head_min_steps) + 1
    assert next_token_id == cfg.text_eos_token_id

    logits = runner.llm.compute_logits(hidden, _DummySamplingMetadata(step=4))
    out = runner.llm.sample(logits, _DummySamplingMetadata(step=4))
    assert int(out.sampled_token_ids[0, 0]) == cfg.text_eos_token_id


def test_ming_compute_logits_uses_forward_stop_prob_payload(monkeypatch):
    runner = _make_runner_for_ming(monkeypatch)
    cfg = runner.config

    def _low_stop(_hidden_states):
        return torch.tensor([[10.0, 0.0]], dtype=torch.float32)

    monkeypatch.setattr(runner.llm.stop_head, "forward", _low_stop)
    hidden = torch.zeros((1, cfg.llm_hidden_size), dtype=torch.float32)

    logits = runner.llm.compute_logits(
        OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={
                "ming_stop_prob": torch.tensor([1.0], dtype=torch.float32),
                "ming_decode_step": torch.tensor([4], dtype=torch.int32),
            },
        ),
        _DummySamplingMetadata(step=4),
    )
    out = runner.llm.sample(logits, _DummySamplingMetadata(step=4))
    assert int(out.sampled_token_ids[0, 0]) == cfg.text_eos_token_id


def test_ming_compute_logits_uses_cached_forward_stop_prob_for_tensor_path(monkeypatch):
    runner = _make_runner_for_ming(monkeypatch)
    cfg = runner.config

    def _low_stop(_hidden_states):
        return torch.tensor([[10.0, 0.0]], dtype=torch.float32)

    monkeypatch.setattr(runner.llm.stop_head, "forward", _low_stop)
    runner.llm._last_sample_stop_probs = torch.tensor([1.0], dtype=torch.float32)
    runner.llm._last_sample_decode_steps = torch.tensor([4], dtype=torch.int32)
    hidden = torch.zeros((1, cfg.llm_hidden_size), dtype=torch.float32)

    logits = runner.llm.compute_logits(hidden, _DummySamplingMetadata(step=4))
    out = runner.llm.sample(logits, _DummySamplingMetadata(step=4))
    assert int(out.sampled_token_ids[0, 0]) == cfg.text_eos_token_id


def test_ming_forward_exposes_stop_reason_in_outputs_and_pending_state(monkeypatch):
    runner = _make_runner_for_ming(monkeypatch)
    cfg = runner.config

    def _low_stop(_hidden_states):
        return torch.tensor([[10.0, 0.0]], dtype=torch.float32)

    monkeypatch.setattr(runner.llm.stop_head, "forward", _low_stop)
    decode_ids = torch.tensor([cfg.audio_dummy_token_id], dtype=torch.long)
    decode_embeds = torch.zeros((1, cfg.llm_hidden_size), dtype=torch.float32)
    output = runner.llm.forward(
        decode_ids,
        positions=torch.tensor([0], dtype=torch.long),
        inputs_embeds=decode_embeds,
        model_intermediate_buffer=[
            {
                KEY_LATENT_HISTORY: torch.zeros((cfg.history_patch_size, cfg.latent_dim), dtype=torch.float32),
                KEY_REQUEST_ID: "req-stop-reason",
            }
        ],
        seq_token_counts=[1],
    )

    assert output.multimodal_outputs[MING_STOP_REASON_KEY] == (MING_STOP_REASON_CONTINUE,)
    pending = runner.llm.pop_postprocess_update("req-stop-reason")
    assert pending[MING_STOP_REASON_KEY] == MING_STOP_REASON_CONTINUE


def test_ming_postprocess_forwards_stop_reason(monkeypatch):
    runner = _make_runner_for_ming(monkeypatch)
    cfg = runner.config

    decode_ids = torch.tensor([cfg.audio_dummy_token_id], dtype=torch.long)
    decode_embeds = torch.zeros((1, cfg.llm_hidden_size), dtype=torch.float32)
    decode_info = {
        KEY_LATENT_HISTORY: torch.zeros((cfg.history_patch_size, cfg.latent_dim), dtype=torch.float32),
        KEY_REQUEST_ID: "req-postprocess-stop-reason",
    }

    output = runner.llm.forward(
        decode_ids,
        positions=torch.tensor([0], dtype=torch.long),
        inputs_embeds=decode_embeds,
        model_intermediate_buffer=[decode_info],
        seq_token_counts=[1],
    )
    update = runner.stage1.postprocess(output.text_hidden_states, **decode_info)

    assert update[MING_STOP_REASON_KEY] == MING_STOP_REASON_CONTINUE


def test_ming_vae_incremental_decode(monkeypatch):
    runner = _make_runner_for_ming(monkeypatch)

    chunk_a = torch.stack(
        [
            torch.ones((4, 64), dtype=torch.float32),
            torch.full((4, 64), 2.0, dtype=torch.float32),
        ],
        dim=0,
    )
    out_a = runner.stage2.forward(
        model_intermediate_buffer=[
            {
                "ming_latent_patches": chunk_a,
                "finished": torch.tensor(False),
                "stream_finished": torch.tensor(False),
                KEY_REQUEST_ID: "r1",
            }
        ]
    )
    wav_a = out_a.multimodal_outputs["model_outputs"][0]
    state_a = runner.vae._stream_state["r1"]
    past_a = runner.vae._past_key_values["r1"]

    chunk_b = torch.full((1, 4, 64), 3.0, dtype=torch.float32)
    out_b = runner.stage2.forward(
        model_intermediate_buffer=[
            {
                "ming_latent_patches": chunk_b,
                "finished": torch.tensor(False),
                "stream_finished": torch.tensor(False),
                KEY_REQUEST_ID: "r1",
            }
        ]
    )
    wav_b = out_b.multimodal_outputs["model_outputs"][0]
    state_b = runner.vae._stream_state["r1"]

    assert len(runner.vae.audio.decode_calls) == 3
    assert runner.vae.audio.decode_calls[1]["latent_shape"] == (1, 4, 64)
    assert runner.vae.audio.decode_calls[1]["past_key_values"] == {"frames": 4}
    assert runner.vae.audio.decode_calls[2]["stream_state"] == state_a
    assert runner.vae.audio.decode_calls[2]["past_key_values"] == past_a
    assert state_b != state_a

    expected_a = torch.cat(
        [
            chunk_a[0].sum(dim=-1),
            chunk_a[1].sum(dim=-1) + 4 * 10.0,
        ]
    )
    expected_b = chunk_b[0].sum(dim=-1) + 8 * 10.0
    assert torch.allclose(wav_a, expected_a)
    assert torch.allclose(wav_b, expected_b)
    assert torch.allclose(torch.cat([wav_a, wav_b]), torch.cat([expected_a, expected_b]))


def test_ming_vae_finalizes_when_stream_finished_is_absent(monkeypatch):
    runner = _make_runner_for_ming(monkeypatch)
    chunk = torch.stack(
        [
            torch.ones((4, 64), dtype=torch.float32),
            torch.full((4, 64), 2.0, dtype=torch.float32),
        ],
        dim=0,
    )

    out = runner.stage2.forward(
        model_intermediate_buffer=[
            {
                "ming_latent_patches": chunk,
                "finished": torch.tensor(True),
                KEY_REQUEST_ID: "r-sequential",
            }
        ]
    )

    wav = out.multimodal_outputs["model_outputs"][0]
    assert wav.numel() > 0
    assert "r-sequential" not in runner.vae._stream_state
    assert "r-sequential" not in runner.vae._past_key_values


def test_ming_recurrent_backbone_can_poison_hidden_states_before_flowloss(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as wrapper_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae as vae_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(
        llm_mod, "init_vllm_registered_model", lambda **kwargs: NaNOnSecondDecodeBackbone(cfg.llm_hidden_size)
    )
    monkeypatch.setattr(llm_mod, "Aggregator", DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", DummyFlowLoss)
    monkeypatch.setattr(vae_mod, "AudioVAE", DummyAudioVAE)

    llm_model = MingLLMModel(vllm_config=_make_vllm_config("llm"))

    def _wrapper_loader(*, architectures, **kwargs):
        arch = architectures[0]
        if arch == "MingLLMModel":
            return llm_model
        raise AssertionError(f"unexpected architecture {arch}")

    monkeypatch.setattr(wrapper_mod, "init_vllm_registered_model", _wrapper_loader)
    stage1 = MingTTSForConditionalGeneration(vllm_config=_make_vllm_config("llm"))

    decode_ids = torch.tensor([cfg.audio_dummy_token_id], dtype=torch.long)
    decode_embeds = torch.zeros((1, cfg.llm_hidden_size), dtype=torch.float32)
    decode_info = {
        KEY_LATENT_HISTORY: torch.zeros((cfg.history_patch_size, cfg.latent_dim), dtype=torch.float32),
        KEY_REQUEST_ID: "req-nan",
    }

    _, decode_embeds, decode_info = stage1.preprocess_input(decode_ids, decode_embeds, **decode_info)
    output = llm_model.forward(
        decode_ids,
        positions=torch.tensor([0], dtype=torch.long),
        inputs_embeds=decode_embeds,
        model_intermediate_buffer=[decode_info],
        seq_token_counts=[1],
    )
    mm = output.multimodal_outputs
    assert torch.isfinite(mm["ming_next_embeds"]).all()

    update = stage1.postprocess(output.text_hidden_states, multimodal_outputs=mm, **decode_info)
    _, next_decode_embeds, next_decode_info = stage1.preprocess_input(
        decode_ids,
        torch.zeros((1, cfg.llm_hidden_size), dtype=torch.float32),
        **update,
    )
    assert torch.isfinite(next_decode_embeds).all()

    with pytest.raises(RuntimeError, match="Non-finite z_diff_cond before FlowLoss.sample"):
        llm_model.forward(
            decode_ids,
            positions=torch.tensor([1], dtype=torch.long),
            inputs_embeds=next_decode_embeds,
            model_intermediate_buffer=[next_decode_info],
            seq_token_counts=[1],
        )
