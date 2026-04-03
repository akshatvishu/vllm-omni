from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming_tts.config_ming_tts import MingTTSConfig
from vllm_omni.model_executor.models.ming_tts.ming_tts import MingTTSForConditionalGeneration
from vllm_omni.model_executor.models.ming_tts.ming_tts_llm import MingLLMModel


class _CapturingStageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_weights: list[tuple[str, torch.Tensor]] = []

    def load_weights(self, weights):
        self.seen_weights = list(weights)
        return {name for name, _ in self.seen_weights}

    def embed_input_ids(self, input_ids, **kwargs):
        del kwargs
        return torch.zeros((input_ids.shape[0], 1), dtype=torch.float32)


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


def _make_vllm_config(model_stage: str):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(), model_stage=model_stage),
        quant_config=None,
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )


def test_ming_wrapper_load_weights_routes_only_llm_prefixes(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as wrapper_mod

    cfg = _make_config()
    llm_model = _CapturingStageModel()
    vae_model = _CapturingStageModel()

    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))

    def _loader(*, architectures, **kwargs):
        del kwargs
        if architectures[0] == "MingLLMModel":
            return llm_model
        if architectures[0] == "MingAudioVAEModel":
            return vae_model
        raise AssertionError(f"unexpected architecture {architectures[0]}")

    monkeypatch.setattr(wrapper_mod, "init_vllm_registered_model", _loader)

    stage1 = MingTTSForConditionalGeneration(vllm_config=_make_vllm_config("llm"))
    monkeypatch.setattr(stage1, "_load_prompt_audio_encoder_weights", lambda weights: None)
    weights = [
        ("model.layers.0.self_attn.q_proj.weight", torch.ones((1,), dtype=torch.float32)),
        ("model.lm_head.weight", torch.full((1,), 2.0, dtype=torch.float32)),
        ("linear_proj_audio.proj_in.weight", torch.full((1,), 3.0, dtype=torch.float32)),
        ("flowloss.dummy.weight", torch.full((1,), 4.0, dtype=torch.float32)),
        ("stop_head.weight", torch.full((1,), 5.0, dtype=torch.float32)),
        ("spk_head.weight", torch.full((1,), 6.0, dtype=torch.float32)),
        ("audio.decoder.weight", torch.full((1,), 7.0, dtype=torch.float32)),
        ("junk.weight", torch.full((1,), 8.0, dtype=torch.float32)),
    ]

    loaded = stage1.load_weights(weights)

    seen_names = [name for name, _ in llm_model.seen_weights]
    assert seen_names == [
        "model.layers.0.self_attn.q_proj.weight",
        "linear_proj_audio.proj_in.weight",
        "flowloss.dummy.weight",
        "stop_head.weight",
        "spk_head.weight",
    ]
    assert "model.lm_head.weight" not in seen_names
    assert "audio.decoder.weight" not in seen_names
    assert "junk.weight" not in seen_names
    assert loaded == {f"model.{name}" for name in seen_names}


def test_ming_wrapper_load_weights_routes_only_audio_prefixes(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as wrapper_mod

    cfg = _make_config()
    llm_model = _CapturingStageModel()
    vae_model = _CapturingStageModel()

    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))

    def _loader(*, architectures, **kwargs):
        del kwargs
        if architectures[0] == "MingLLMModel":
            return llm_model
        if architectures[0] == "MingAudioVAEModel":
            return vae_model
        raise AssertionError(f"unexpected architecture {architectures[0]}")

    monkeypatch.setattr(wrapper_mod, "init_vllm_registered_model", _loader)

    stage2 = MingTTSForConditionalGeneration(vllm_config=_make_vllm_config("audio_vae"))
    weights = [
        ("audio.decoder.weight", torch.ones((1,), dtype=torch.float32)),
        ("audio.encoder.weight", torch.full((1,), 2.0, dtype=torch.float32)),
        ("model.layers.0.self_attn.q_proj.weight", torch.full((1,), 3.0, dtype=torch.float32)),
        ("flowloss.dummy.weight", torch.full((1,), 4.0, dtype=torch.float32)),
    ]

    loaded = stage2.load_weights(weights)

    seen_names = [name for name, _ in vae_model.seen_weights]
    assert seen_names == [
        "audio.decoder.weight",
        "audio.encoder.weight",
    ]
    assert loaded == {f"model.{name}" for name in seen_names}


def test_ming_wrapper_load_weights_rejects_empty_stage_inputs(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts as wrapper_mod

    cfg = _make_config()
    llm_model = _CapturingStageModel()
    vae_model = _CapturingStageModel()

    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))

    def _loader(*, architectures, **kwargs):
        del kwargs
        if architectures[0] == "MingLLMModel":
            return llm_model
        if architectures[0] == "MingAudioVAEModel":
            return vae_model
        raise AssertionError(f"unexpected architecture {architectures[0]}")

    monkeypatch.setattr(wrapper_mod, "init_vllm_registered_model", _loader)

    stage1 = MingTTSForConditionalGeneration(vllm_config=_make_vllm_config("llm"))
    stage2 = MingTTSForConditionalGeneration(vllm_config=_make_vllm_config("audio_vae"))

    with pytest.raises(RuntimeError, match="Stage-1 received no loadable checkpoint weights"):
        stage1.load_weights([("junk.weight", torch.ones((1,), dtype=torch.float32))])

    with pytest.raises(RuntimeError, match="Stage-2 received no loadable checkpoint weights"):
        stage2.load_weights([("junk.weight", torch.ones((1,), dtype=torch.float32))])


class _DummySelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(1, 3, bias=True)
        self.o_proj = torch.nn.Linear(1, 1, bias=False)


class _DummyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(1, 2, bias=False)
        self.down_proj = torch.nn.Linear(2, 1, bias=False)


class _DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _DummySelfAttention()
        self.mlp = _DummyMLP()
        self.input_layernorm = torch.nn.LayerNorm(1)
        self.post_attention_layernorm = torch.nn.LayerNorm(1)


class _DummyQwenForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(4, 1)
        self.model.layers = torch.nn.ModuleList([_DummyLayer()])
        self.model.norm = torch.nn.LayerNorm(1)


class _DummyAggregator(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.proj = torch.nn.Linear(1, 1, bias=False)


class _DummyFlowLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.dummy = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32))


def _install_shard_recording_loader(
    param: torch.nn.Parameter,
    *,
    shard_to_index: dict[object, int],
    calls: list[tuple[object, torch.Tensor]],
) -> None:
    def _loader(target_param, loaded_weight, shard_id):
        calls.append((shard_id, loaded_weight.detach().clone()))
        row = shard_to_index[shard_id]
        if target_param.ndim == 2:
            target_param.data[row : row + 1].copy_(loaded_weight.to(dtype=target_param.dtype))
        else:
            target_param.data[row : row + 1].copy_(loaded_weight.reshape(1).to(dtype=target_param.dtype))

    param.weight_loader = _loader


def _make_llm_vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(), dtype=torch.float32),
        quant_config=None,
    )


def test_ming_stage0_loader_keeps_model_model_prefix_without_extra_model(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()

    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: _DummyQwenForCausalLM())
    monkeypatch.setattr(llm_mod, "Aggregator", _DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", _DummyFlowLoss)
    monkeypatch.setattr(llm_mod, "_warn_missing_prefix", lambda *args, **kwargs: None)

    model = MingLLMModel(vllm_config=_make_llm_vllm_config())
    qkv_weight = torch.full((3, 1), 7.0, dtype=torch.float32)

    loaded = model.load_weights(
        [
            ("model.model.layers.0.self_attn.q_proj.weight", qkv_weight),
        ]
    )

    assert "model.model.layers.0.self_attn.qkv_proj.weight" in loaded
    assert "model.model.model.layers.0.self_attn.qkv_proj.weight" not in loaded
    assert torch.allclose(model.model.model.layers[0].self_attn.qkv_proj.weight[:, :1], qkv_weight)


def test_ming_stage0_loader_routes_qkv_shards_through_target_param_weight_loader(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()

    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: _DummyQwenForCausalLM())
    monkeypatch.setattr(llm_mod, "Aggregator", _DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", _DummyFlowLoss)
    monkeypatch.setattr(llm_mod, "_warn_missing_prefix", lambda *args, **kwargs: None)

    model = MingLLMModel(vllm_config=_make_llm_vllm_config())
    qkv_weight_calls: list[tuple[object, torch.Tensor]] = []
    qkv_bias_calls: list[tuple[object, torch.Tensor]] = []
    qkv_proj = model.model.model.layers[0].self_attn.qkv_proj
    _install_shard_recording_loader(
        qkv_proj.weight,
        shard_to_index={"q": 0, "k": 1, "v": 2},
        calls=qkv_weight_calls,
    )
    _install_shard_recording_loader(
        qkv_proj.bias,
        shard_to_index={"q": 0, "k": 1, "v": 2},
        calls=qkv_bias_calls,
    )

    loaded = model.load_weights(
        [
            ("model.model.layers.0.self_attn.q_proj.weight", torch.tensor([[11.0]], dtype=torch.float32)),
            ("model.model.layers.0.self_attn.k_proj.weight", torch.tensor([[22.0]], dtype=torch.float32)),
            ("model.model.layers.0.self_attn.v_proj.weight", torch.tensor([[33.0]], dtype=torch.float32)),
            ("model.model.layers.0.self_attn.q_proj.bias", torch.tensor([1.0], dtype=torch.float32)),
            ("model.model.layers.0.self_attn.k_proj.bias", torch.tensor([2.0], dtype=torch.float32)),
            ("model.model.layers.0.self_attn.v_proj.bias", torch.tensor([3.0], dtype=torch.float32)),
        ]
    )

    assert [shard_id for shard_id, _ in qkv_weight_calls] == ["q", "k", "v"]
    assert [shard_id for shard_id, _ in qkv_bias_calls] == ["q", "k", "v"]
    assert "model.model.layers.0.self_attn.qkv_proj.weight" in loaded
    assert "model.model.layers.0.self_attn.qkv_proj.bias" in loaded
    assert "model.model.model.layers.0.self_attn.qkv_proj.weight" not in loaded
    assert "model.model.model.layers.0.self_attn.qkv_proj.bias" not in loaded
    assert torch.allclose(qkv_proj.weight[:, :1], torch.tensor([[11.0], [22.0], [33.0]], dtype=torch.float32))
    assert torch.allclose(qkv_proj.bias, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))


def test_ming_stage0_loader_routes_gate_up_shards_through_target_param_weight_loader(monkeypatch):
    import vllm_omni.model_executor.models.ming_tts.config_ming_tts as cfg_mod
    import vllm_omni.model_executor.models.ming_tts.ming_tts_llm as llm_mod

    cfg = _make_config()

    monkeypatch.setattr(cfg_mod.MingTTSConfig, "from_hf_config", classmethod(lambda cls, hf: cfg))
    monkeypatch.setattr(llm_mod, "init_vllm_registered_model", lambda **kwargs: _DummyQwenForCausalLM())
    monkeypatch.setattr(llm_mod, "Aggregator", _DummyAggregator)
    monkeypatch.setattr(llm_mod, "FlowLoss", _DummyFlowLoss)
    monkeypatch.setattr(llm_mod, "_warn_missing_prefix", lambda *args, **kwargs: None)

    model = MingLLMModel(vllm_config=_make_llm_vllm_config())
    gate_up_calls: list[tuple[object, torch.Tensor]] = []
    gate_up_proj = model.model.model.layers[0].mlp.gate_up_proj
    _install_shard_recording_loader(
        gate_up_proj.weight,
        shard_to_index={0: 0, 1: 1},
        calls=gate_up_calls,
    )

    loaded = model.load_weights(
        [
            ("model.model.layers.0.mlp.gate_proj.weight", torch.tensor([[5.0]], dtype=torch.float32)),
            ("model.model.layers.0.mlp.up_proj.weight", torch.tensor([[7.0]], dtype=torch.float32)),
        ]
    )

    assert [shard_id for shard_id, _ in gate_up_calls] == [0, 1]
    assert "model.model.layers.0.mlp.gate_up_proj.weight" in loaded
    assert "model.model.model.layers.0.mlp.gate_up_proj.weight" not in loaded
    assert torch.allclose(gate_up_proj.weight[:, :1], torch.tensor([[5.0], [7.0]], dtype=torch.float32))
