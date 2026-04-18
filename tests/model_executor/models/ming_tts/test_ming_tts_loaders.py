# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.outputs import SamplerOutput

from vllm_omni.model_executor.models.ming_tts.config_ming_tts import KEY_PROMPT_LATENTS, KEY_REQUEST_ID, MingTTSConfig
from vllm_omni.model_executor.models.ming_tts.ming_tts import MingTTSForConditionalGeneration
from vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae import MingAudioVAEModel
from vllm_omni.model_executor.models.ming_tts.ming_tts_llm import MingLLMModel


class _DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
        self.last_forward_kwargs = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], 2), dtype=torch.float32)

    def forward(self, *args, **kwargs):
        del args
        self.last_forward_kwargs = dict(kwargs)
        return torch.zeros((1, 2), dtype=torch.float32)


class _DummyAggregator(torch.nn.Module):
    def __init__(self, in_channels: int, llm_input_dim: int, **kwargs):
        super().__init__()
        del kwargs
        self.proj_in = torch.nn.Linear(in_channels, llm_input_dim, bias=False)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.proj_in(patch.mean(dim=1)).unsqueeze(1)


class _DummyFlowLoss(torch.nn.Module):
    def __init__(self, z_channels: int, llm_cond_dim: int, **kwargs):
        super().__init__()
        del z_channels, kwargs
        self.dummy = torch.nn.Linear(llm_cond_dim, 64, bias=False)

    def sample(self, **kwargs):
        del kwargs
        return torch.zeros((1, 4, 64), dtype=torch.float32)


class _DummyAudioVAE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        del config
        self.encoder = torch.nn.Linear(2, 2, bias=False)
        self.decoder = torch.nn.Linear(2, 2, bias=False)
        self.last_chunk_values = []

    def encode_latent(self, waveform: torch.Tensor, waveform_length: torch.Tensor):
        del waveform_length
        batch = int(waveform.shape[0])
        return torch.zeros((batch, 8, 64), dtype=torch.float32), None

    def decode(
        self,
        latent_patch: torch.Tensor,
        *,
        past_key_values=None,
        use_cache=True,
        stream_state=None,
        last_chunk=False,
    ):
        del past_key_values, use_cache, stream_state
        self.last_chunk_values.append(last_chunk)
        samples = int(latent_patch.shape[1]) * 8
        waveform = torch.ones((1, 1, samples), dtype=torch.float32)
        return waveform, (None, None, None), None


def _make_audio_cfg():
    return SimpleNamespace(
        enc_kwargs={
            "backbone": {"hidden_size": 2},
            "input_dim": 882,
            "hop_size": 882,
            "latent_dim": 64,
        },
        dec_kwargs={
            "backbone": {"hidden_size": 2},
            "output_dim": 882,
            "latent_dim": 64,
        },
        patch_size=4,
        sample_rate=44100,
        semantic_module_kwargs=None,
    )


def _make_config() -> MingTTSConfig:
    cfg = MingTTSConfig(audio_tokenizer_config=_make_audio_cfg())
    cfg.validate()
    return cfg


def _make_vllm_config(model_stage: str):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(), model_stage=model_stage),
        quant_config=None,
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )


def test_ming_llm_load_weights_maps_and_loads_expected_prefixes(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: _DummyBackbone())
    monkeypatch.setattr(llm_mod, "Aggregator", _DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", _DummyFlowLoss)

    model = MingLLMModel(vllm_config=_make_vllm_config("llm"))
    weights = [
        ("model.model.layers.0.weight", torch.full((2, 2), 1.0, dtype=torch.float32)),
        ("linear_proj_audio.proj_in.weight", torch.full((896, 64), 2.0, dtype=torch.float32)),
        ("flowloss.dummy.weight", torch.full((64, 896), 3.0, dtype=torch.float32)),
        ("stop_head.weight", torch.full((2, 896), 4.0, dtype=torch.float32)),
        ("stop_head.bias", torch.full((2,), 5.0, dtype=torch.float32)),
        ("spk_head.weight", torch.full((896, 192), 6.0, dtype=torch.float32)),
        ("spk_head.bias", torch.full((896,), 7.0, dtype=torch.float32)),
    ]

    loaded = model.load_weights(weights)

    assert "model.model.layers.0.weight" in loaded
    assert "linear_proj_audio.proj_in.weight" in loaded
    assert "flowloss.dummy.weight" in loaded
    assert "stop_head.weight" in loaded
    assert "spk_head.weight" in loaded
    assert torch.allclose(model.model.model.layers[0].weight, torch.full((2, 2), 1.0))
    assert torch.allclose(model.linear_proj_audio.proj_in.weight, torch.full((896, 64), 2.0))
    assert torch.allclose(model.flowloss.dummy.weight, torch.full((64, 896), 3.0))


def test_ming_llm_load_weights_accepts_complete_checkpoint_and_forward_shape(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: _DummyBackbone())
    monkeypatch.setattr(llm_mod, "Aggregator", _DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", _DummyFlowLoss)

    model = MingLLMModel(vllm_config=_make_vllm_config("llm"))
    model.load_weights(
        [
            ("model.layers.0.weight", torch.ones((2, 2), dtype=torch.float32)),
            ("linear_proj_audio.proj_in.weight", torch.ones((896, 64), dtype=torch.float32)),
            ("flowloss.dummy.weight", torch.ones((64, 896), dtype=torch.float32)),
            ("stop_head.weight", torch.ones((2, 896), dtype=torch.float32)),
            ("stop_head.bias", torch.ones((2,), dtype=torch.float32)),
            ("spk_head.weight", torch.ones((896, 192), dtype=torch.float32)),
            ("spk_head.bias", torch.ones((896,), dtype=torch.float32)),
        ]
    )

    output = model.forward(
        input_ids=torch.tensor([1], dtype=torch.long),
        positions=torch.tensor([0], dtype=torch.long),
    )

    assert output.text_hidden_states.shape == (1, 2)
    assert output.multimodal_outputs is None


def test_ming_llm_load_weights_fails_when_custom_heads_missing(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: _DummyBackbone())
    monkeypatch.setattr(llm_mod, "Aggregator", _DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", _DummyFlowLoss)

    model = MingLLMModel(vllm_config=_make_vllm_config("llm"))
    weights = [
        ("model.layers.0.weight", torch.full((2, 2), 1.0, dtype=torch.float32)),
        ("stop_head.weight", torch.full((2, 896), 4.0, dtype=torch.float32)),
        ("stop_head.bias", torch.full((2,), 5.0, dtype=torch.float32)),
        ("spk_head.weight", torch.full((896, 192), 6.0, dtype=torch.float32)),
        ("spk_head.bias", torch.full((896,), 7.0, dtype=torch.float32)),
    ]

    with pytest.raises(RuntimeError, match="flowloss|linear_proj_audio"):
        model.load_weights(weights)


def test_ming_llm_load_weights_rejects_incomplete_checkpoint(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: _DummyBackbone())
    monkeypatch.setattr(llm_mod, "Aggregator", _DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", _DummyFlowLoss)

    model = MingLLMModel(vllm_config=_make_vllm_config("llm"))

    with pytest.raises(RuntimeError, match="flowloss|linear_proj_audio|stop_head|spk_head"):
        model.load_weights(
            [
                ("model.layers.0.weight", torch.ones((2, 2), dtype=torch.float32)),
                ("stop_head.weight", torch.ones((2, 896), dtype=torch.float32)),
                ("stop_head.bias", torch.ones((2,), dtype=torch.float32)),
            ]
        )


def test_ming_audio_vae_load_weights_fails_when_audio_params_missing(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae as vae_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(vae_mod, "AudioVAE", _DummyAudioVAE)

    model = MingAudioVAEModel(vllm_config=_make_vllm_config("audio_vae"))

    with pytest.raises(RuntimeError, match="params not loaded"):
        model.load_weights(
            [
                ("audio.encoder.weight", torch.full((2, 2), 1.0, dtype=torch.float32)),
            ]
        )


def test_ming_audio_vae_load_weights_accepts_complete_checkpoint_and_forward_shape(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae as vae_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(vae_mod, "AudioVAE", _DummyAudioVAE)

    model = MingAudioVAEModel(vllm_config=_make_vllm_config("audio_vae"))
    model.load_weights(
        [
            ("audio.encoder.weight", torch.ones((2, 2), dtype=torch.float32)),
            ("audio.decoder.weight", torch.ones((2, 2), dtype=torch.float32)),
        ]
    )

    output = model.forward(
        runtime_additional_information=[
            {
                KEY_REQUEST_ID: "rid-audio",
                "ming_latent_patches": torch.ones((1, 4, 64), dtype=torch.float32),
                "stream_finished": torch.tensor(True, dtype=torch.bool),
            }
        ]
    )

    waveform = output.multimodal_outputs["model_outputs"][0]
    sample_rate = output.multimodal_outputs["sr"][0]
    assert waveform.ndim == 1
    assert waveform.dtype == torch.float32
    assert waveform.shape == (32,)
    assert int(sample_rate.item()) == 44100
    assert model.audio.last_chunk_values == [True]


def test_ming_audio_vae_load_weights_rejects_incomplete_checkpoint(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae as vae_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(vae_mod, "AudioVAE", _DummyAudioVAE)

    model = MingAudioVAEModel(vllm_config=_make_vllm_config("audio_vae"))

    with pytest.raises(RuntimeError, match="params not loaded|no checkpoint weights"):
        model.load_weights(
            [
                ("audio.encoder.weight", torch.ones((2, 2), dtype=torch.float32)),
            ]
        )


def test_ming_audio_vae_load_weights_rejects_empty_input(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae as vae_mod

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(vae_mod, "AudioVAE", _DummyAudioVAE)

    model = MingAudioVAEModel(vllm_config=_make_vllm_config("audio_vae"))

    with pytest.raises(RuntimeError, match="no checkpoint weights"):
        model.load_weights([])


def test_ming_llm_forward_drops_runner_only_kwargs(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()
    backbone = _DummyBackbone()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: backbone)
    monkeypatch.setattr(llm_mod, "Aggregator", _DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", _DummyFlowLoss)

    model = MingLLMModel(vllm_config=_make_vllm_config("llm"))
    output = model.forward(
        input_ids=torch.tensor([1], dtype=torch.long),
        positions=torch.tensor([0], dtype=torch.long),
        sampling_metadata=object(),
        logits_index=0,
        sampler=object(),
        additional_information={"text": "hello"},
    )

    assert set(backbone.last_forward_kwargs) == {
        "input_ids",
        "positions",
        "intermediate_tensors",
        "inputs_embeds",
    }
    assert torch.equal(backbone.last_forward_kwargs["input_ids"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(backbone.last_forward_kwargs["positions"], torch.tensor([0], dtype=torch.long))
    assert backbone.last_forward_kwargs["intermediate_tensors"] is None
    assert torch.allclose(backbone.last_forward_kwargs["inputs_embeds"], torch.zeros((1, 2), dtype=torch.float32))
    assert output.text_hidden_states.shape == (1, 2)
    assert output.multimodal_outputs is None


def test_ming_llm_forward_normalizes_runtime_additional_information(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()
    backbone = _DummyBackbone()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: backbone)
    monkeypatch.setattr(llm_mod, "Aggregator", _DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", _DummyFlowLoss)

    model = MingLLMModel(vllm_config=_make_vllm_config("llm"))
    output = model.forward(
        input_ids=torch.tensor([1], dtype=torch.long),
        positions=torch.tensor([0], dtype=torch.long),
        runtime_additional_information=[{"decode_step": 0}],
    )

    assert set(backbone.last_forward_kwargs) == {
        "input_ids",
        "positions",
        "intermediate_tensors",
        "inputs_embeds",
    }
    assert torch.equal(backbone.last_forward_kwargs["input_ids"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(backbone.last_forward_kwargs["positions"], torch.tensor([0], dtype=torch.long))
    assert backbone.last_forward_kwargs["intermediate_tensors"] is None
    assert torch.allclose(backbone.last_forward_kwargs["inputs_embeds"], torch.zeros((1, 2), dtype=torch.float32))
    assert output.text_hidden_states.shape == (1, 2)
    assert output.multimodal_outputs is None


def test_ming_stage0_sampler_uses_model_sample(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as ming_mod

    class _DummyStage0(torch.nn.Module):
        def sample(self, logits, sampling_metadata):
            del logits, sampling_metadata
            return SamplerOutput(
                sampled_token_ids=torch.tensor([[151705]], dtype=torch.int32),
                logprobs_tensors=None,
            )

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(ming_mod, "init_vllm_registered_model", lambda **kwargs: _DummyStage0())

    model = MingTTSForConditionalGeneration(vllm_config=_make_vllm_config("llm"))
    sampler_output = model.sampler(
        torch.zeros((1, cfg.llm_vocab_size), dtype=torch.float32),
        SimpleNamespace(seq_groups=[]),
    )

    assert isinstance(sampler_output, SamplerOutput)
    assert sampler_output.sampled_token_ids.dtype == torch.int32
    assert sampler_output.sampled_token_ids.tolist() == [[151705]]


def test_ming_stage0_load_weights_does_not_load_audio_weights(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as ming_mod

    class _DummyStage0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded = None

        def load_weights(self, weights):
            self.loaded = list(weights)
            return {name for name, _ in self.loaded}

    cfg = _make_config()
    stage0 = _DummyStage0()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(ming_mod, "init_vllm_registered_model", lambda **kwargs: stage0)

    model = MingTTSForConditionalGeneration(vllm_config=_make_vllm_config("llm"))
    loaded = model.load_weights(
        [
            ("model.layers.0.weight", torch.ones((2, 2), dtype=torch.float32)),
            ("linear_proj_audio.proj_in.weight", torch.ones((896, 64), dtype=torch.float32)),
            ("flowloss.dummy.weight", torch.ones((64, 896), dtype=torch.float32)),
            ("stop_head.weight", torch.ones((2, 896), dtype=torch.float32)),
            ("spk_head.weight", torch.ones((896, 192), dtype=torch.float32)),
            ("audio.encoder.weight", torch.ones((2, 2), dtype=torch.float32)),
        ]
    )

    assert "model.audio.encoder.weight" not in loaded
    assert all(not name.startswith("audio.") for name, _ in stage0.loaded)
    assert not hasattr(model, "_prompt_audio_encoder")


def test_ming_resolve_prompt_latents_accepts_raw_waveform(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as ming_mod

    class _DummyStage0(torch.nn.Module):
        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros((input_ids.shape[0], 2), dtype=torch.float32)

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(ming_mod, "init_vllm_registered_model", lambda **kwargs: _DummyStage0())

    model = MingTTSForConditionalGeneration(vllm_config=_make_vllm_config("llm"))
    direct = torch.ones((8, 64), dtype=torch.float32)

    resolved = model._resolve_prompt_latents({KEY_PROMPT_LATENTS: direct})
    assert resolved is not None
    assert torch.equal(resolved["frames"], direct)

    model._encode_prompt_waveform_to_latents = lambda waveform, waveform_length=None: torch.ones(
        (8, 64), dtype=torch.float32
    )
    resolved = model._resolve_prompt_latents(
        {
            "prompt_waveform": torch.ones((1, 1000), dtype=torch.float32),
            "prompt_waveform_length": torch.tensor([1000], dtype=torch.int32),
            "prompt_text": "Reference words.",
        }
    )
    assert resolved is not None
    assert resolved["patches"].shape == (2, 4, 64)


def test_ming_resolve_prompt_latents_rejects_dual_truth_waveform_and_latents(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as ming_mod

    class _DummyStage0(torch.nn.Module):
        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros((input_ids.shape[0], 2), dtype=torch.float32)

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(ming_mod, "init_vllm_registered_model", lambda **kwargs: _DummyStage0())

    model = MingTTSForConditionalGeneration(vllm_config=_make_vllm_config("llm"))

    with pytest.raises(ValueError, match="Choose exactly one source of truth"):
        model._resolve_prompt_latents(
            {
                KEY_PROMPT_LATENTS: torch.ones((8, 64), dtype=torch.float32),
                "prompt_waveform": torch.ones((1, 1000), dtype=torch.float32),
                "prompt_waveform_length": torch.tensor([1000], dtype=torch.int32),
                "prompt_text": "Reference words.",
            }
        )


def test_ming_prefill_overwrites_speaker_slot_embedding(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as ming_mod

    class _DummyStage0(torch.nn.Module):
        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.arange(int(input_ids.shape[0]) * 2, dtype=torch.float32).reshape(int(input_ids.shape[0]), 2)

        def project_speaker_embedding(self, spk_emb: torch.Tensor) -> torch.Tensor:
            del spk_emb
            return torch.tensor([[101.0, 202.0]], dtype=torch.float32)

    cfg = _make_config()
    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(ming_mod, "init_vllm_registered_model", lambda **kwargs: _DummyStage0())

    vllm_config = _make_vllm_config("llm")
    vllm_config.model_config.hf_config = SimpleNamespace(vision_start_token_id=10)
    model = MingTTSForConditionalGeneration(vllm_config=vllm_config)

    input_ids = torch.tensor([1, 10, 20, 2], dtype=torch.long)
    input_embeds = model.model.embed_input_ids(input_ids)
    _, updated_embeds, _ = model._prefill_preprocess(
        input_ids,
        input_embeds,
        speaker_embedding=torch.ones((192,), dtype=torch.float32),
    )

    assert torch.allclose(updated_embeds[2], torch.tensor([101.0, 202.0], dtype=torch.float32))
