# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter as diffusers_adapter
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class _FakeLoadedPipeline:
    def __init__(self):
        self.device = None

    def __call__(self, prompt=None):
        return prompt

    def to(self, device):
        self.device = device
        return self


class _FakeSingleFilePipeline:
    calls = []

    @classmethod
    def from_single_file(cls, model_id, **kwargs):
        cls.calls.append(("from_single_file", model_id, kwargs))
        return _FakeLoadedPipeline()

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append(("from_pretrained", model_id, kwargs))
        return _FakeLoadedPipeline()


class _FakeNoSingleFilePipeline:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return _FakeLoadedPipeline()


def _make_adapter_for_load_weights(
    *,
    model,
    diffusion_load_format,
    pipeline_cls=None,
    diffusers_load_kwargs=None,
):
    adapter = DiffusersAdapterPipeline.__new__(DiffusersAdapterPipeline)
    adapter.od_config = SimpleNamespace(
        model=model,
        dtype=torch.float32,
        diffusers_load_kwargs=diffusers_load_kwargs or {},
        diffusers_pipeline_cls=pipeline_cls,
        diffusion_load_format=diffusion_load_format,
        enable_layerwise_offload=False,
        enable_cpu_offload=False,
        vae_use_slicing=False,
        vae_use_tiling=False,
    )
    adapter.device = torch.device("cpu")
    adapter._set_attention_backend = lambda: None
    return adapter


def test_anima_registration():
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    assert DiffusionModelRegistry._try_load_model_cls("AnimaPipeline") is not None


def test_enrich_config_single_file(tmp_path):
    dummy_checkpoint = tmp_path / "model.safetensors"
    dummy_checkpoint.write_text("dummy")

    config = OmniDiffusionConfig(
        model=str(dummy_checkpoint),
        diffusion_load_format="diffusers_single_file",
        model_class_name="AnimaModularPipeline",
    )
    config.enrich_config()

    assert config.diffusion_load_format == "default"
    assert config.model_class_name == "AnimaPipeline"
    assert config.diffusers_pipeline_cls is None


def test_enrich_config_single_file_autodetects_local_file(tmp_path):
    dummy_checkpoint = tmp_path / "model.safetensors"
    dummy_checkpoint.write_text("dummy")

    config = OmniDiffusionConfig(
        model=str(dummy_checkpoint),
        model_class_name="AnimaModularPipeline",
    )
    config.enrich_config()

    assert config.diffusion_load_format == "default"
    assert config.model_class_name == "AnimaPipeline"
    assert config.diffusers_pipeline_cls is None


def test_enrich_config_native_anima_single_file_stays_native(tmp_path):
    dummy_checkpoint = tmp_path / "model.safetensors"
    dummy_checkpoint.write_text("dummy")

    config = OmniDiffusionConfig(
        model=str(dummy_checkpoint),
        model_class_name="AnimaPipeline",
    )
    config.enrich_config()

    assert config.diffusion_load_format == "default"
    assert config.model_class_name == "AnimaPipeline"
    assert config.diffusers_pipeline_cls is None


def test_native_anima_single_file_allows_load_kwargs(tmp_path):
    dummy_checkpoint = tmp_path / "model.safetensors"
    dummy_checkpoint.write_text("dummy")

    config = OmniDiffusionConfig(
        model=str(dummy_checkpoint),
        model_class_name="AnimaPipeline",
        diffusers_load_kwargs={"local_files_only": True},
    )
    config.enrich_config()

    assert config.diffusion_load_format == "default"
    assert config.diffusers_load_kwargs == {"local_files_only": True}


def test_diffusers_adapter_accepts_var_keyword_call_signature():
    class ModularLikePipeline:
        def __call__(self, state=None, output=None, **kwargs):
            return kwargs

    pipeline = DiffusersAdapterPipeline.__new__(DiffusersAdapterPipeline)
    pipeline._pipeline = ModularLikePipeline()
    pipeline._accept_call_kwargs = DiffusersAdapterPipeline._get_accepted_call_kwargs(pipeline._pipeline.__call__)
    pipeline._pipeline_utils = SimpleNamespace(validate_runtime_sampling_params=lambda sampling: None)
    pipeline.od_config = SimpleNamespace(
        diffusers_call_kwargs={"height": 512},
        output_type="pil",
    )

    req = OmniDiffusionRequest(
        prompts=["a red cube"],
        sampling_params=OmniDiffusionSamplingParams(
            width=768,
            num_inference_steps=4,
            num_outputs_per_prompt=2,
            seed=123,
            generator_device="cpu",
        ),
        request_id="req",
    )

    kwargs = pipeline._build_call_kwargs(req)

    assert kwargs["prompt"] == "a red cube"
    assert kwargs["height"] == 512
    assert kwargs["width"] == 768
    assert kwargs["num_inference_steps"] == 4
    assert kwargs["num_images_per_prompt"] == 2


def test_diffusers_adapter_loads_explicit_single_file_format(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_text("dummy")
    _FakeSingleFilePipeline.calls = []

    adapter = _make_adapter_for_load_weights(
        model=str(checkpoint),
        diffusion_load_format="diffusers_single_file",
        pipeline_cls=_FakeSingleFilePipeline,
        diffusers_load_kwargs={"custom": "value"},
    )
    adapter.load_weights()

    assert len(_FakeSingleFilePipeline.calls) == 1
    method, model_id, kwargs = _FakeSingleFilePipeline.calls[0]
    assert method == "from_single_file"
    assert model_id == str(checkpoint)
    assert kwargs["torch_dtype"] is torch.float32
    assert kwargs["custom"] == "value"
    assert adapter._pipeline.device == torch.device("cpu")


def test_diffusers_adapter_autodetects_local_single_file(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_text("dummy")
    _FakeSingleFilePipeline.calls = []

    adapter = _make_adapter_for_load_weights(
        model=str(checkpoint),
        diffusion_load_format="diffusers",
        pipeline_cls=_FakeSingleFilePipeline,
    )
    adapter.load_weights()

    assert _FakeSingleFilePipeline.calls[0][0] == "from_single_file"


def test_diffusers_adapter_repo_layout_uses_from_pretrained(monkeypatch):
    calls = []

    def fake_from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return _FakeLoadedPipeline()

    monkeypatch.setattr(diffusers_adapter.DiffusionPipeline, "from_pretrained", staticmethod(fake_from_pretrained))
    adapter = _make_adapter_for_load_weights(
        model="repo/model",
        diffusion_load_format="diffusers",
        pipeline_cls=_FakeSingleFilePipeline,
    )
    adapter.load_weights()

    assert len(calls) == 1
    assert calls[0][0] == "repo/model"
    assert calls[0][1]["torch_dtype"] is torch.float32


def test_diffusers_adapter_single_file_requires_supported_pipeline(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_text("dummy")

    adapter = _make_adapter_for_load_weights(
        model=str(checkpoint),
        diffusion_load_format="diffusers_single_file",
        pipeline_cls=_FakeNoSingleFilePipeline,
    )

    with pytest.raises(ValueError, match="does not support from_single_file"):
        adapter.load_weights()


def test_native_anima_converts_original_cosmos_transformer_keys():
    from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline

    converted = AnimaPipeline._convert_original_transformer_state_dict(
        {
            "net.x_embedder.proj.1.weight": "patch",
            "net.blocks.0.self_attn.q_proj.weight": "q",
            "net.blocks.0.self_attn.q_norm.weight": "q_norm",
            "net.blocks.0.mlp.layer1.weight": "mlp",
            "net.final_layer.linear.weight": "out",
            "net.accum_iteration": "drop",
        }
    )

    assert converted == {
        "patch_embed.proj.weight": "patch",
        "transformer_blocks.0.attn1.to_q.weight": "q",
        "transformer_blocks.0.attn1.norm_q.weight": "q_norm",
        "transformer_blocks.0.ff.net.0.proj.weight": "mlp",
        "proj_out.weight": "out",
    }


def test_native_anima_resolves_vae_scale_factor_from_loaded_vae():
    from vllm_omni.diffusion.models.anima.pipeline_anima import _anima_vae_scale_factor_from_vae

    vae = SimpleNamespace(config=SimpleNamespace(spatial_compression_ratio=16))

    assert _anima_vae_scale_factor_from_vae(vae) == 16


def test_native_anima_loads_synthetic_single_file(tmp_path, monkeypatch):
    import vllm_omni.diffusion.models.anima.pipeline_anima as pipeline_anima
    from vllm_omni.diffusion.models.anima.anima_text_conditioner import AnimaTextConditioner
    from vllm_omni.diffusion.models.anima.anima_transformer import AnimaTransformer3DModel

    tiny_transformer_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_attention_heads": 1,
        "attention_head_dim": 12,
        "num_layers": 1,
        "mlp_ratio": 1.0,
        "text_embed_dim": 4,
        "adaln_lora_dim": 3,
        "max_size": (1, 2, 2),
        "patch_size": (1, 1, 1),
        "rope_scale": (1.0, 1.0, 1.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": None,
    }
    tiny_text_conditioner_config = {
        "source_dim": 4,
        "target_dim": 4,
        "model_dim": 4,
        "num_layers": 1,
        "num_attention_heads": 1,
        "target_vocab_size": 8,
        "min_sequence_length": 4,
    }
    monkeypatch.setattr(pipeline_anima, "ANIMA_TRANSFORMER_CONFIG", tiny_transformer_config)
    monkeypatch.setattr(pipeline_anima, "ANIMA_TEXT_CONDITIONER_CONFIG", tiny_text_conditioner_config)

    transformer = AnimaTransformer3DModel(**tiny_transformer_config)
    text_conditioner = AnimaTextConditioner(**tiny_text_conditioner_config)
    transformer_state = {name: tensor.detach().clone() for name, tensor in transformer.state_dict().items()}
    text_conditioner_state = {name: tensor.detach().clone() for name, tensor in text_conditioner.state_dict().items()}
    checkpoint_state = {
        **{f"transformer.{name}": tensor for name, tensor in transformer_state.items()},
        **{f"text_conditioner.{name}": tensor for name, tensor in text_conditioner_state.items()},
    }

    checkpoint_path = tmp_path / "anima.safetensors"
    save_file(checkpoint_state, str(checkpoint_path))

    pipeline = pipeline_anima.AnimaPipeline.__new__(pipeline_anima.AnimaPipeline)
    pipeline.od_config = SimpleNamespace(model=str(checkpoint_path), dtype=torch.float32)
    pipeline.device = torch.device("cpu")

    def assert_loaded(loaded_transformer, loaded_text_conditioner):
        for name, tensor in transformer_state.items():
            assert torch.equal(loaded_transformer.state_dict()[name], tensor)
        for name, tensor in text_conditioner_state.items():
            assert torch.equal(loaded_text_conditioner.state_dict()[name], tensor)

    loaded_transformer, loaded_text_conditioner = pipeline._load_native_denoiser_components(dict(checkpoint_state))
    assert_loaded(loaded_transformer, loaded_text_conditioner)

    loaded_transformer, loaded_text_conditioner = pipeline._load_native_denoiser_components()
    assert_loaded(loaded_transformer, loaded_text_conditioner)


def _make_anima_forward_probe():
    from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline

    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor = 8
    pipeline.transformer = SimpleNamespace(dtype=torch.float32)
    pipeline.text_encoder = SimpleNamespace(dtype=torch.float32)
    pipeline._current_timestep = None
    pipeline._num_timesteps = 0
    pipeline._guidance_scale = 0.0
    captured = {}

    def encode_prompt(**kwargs):
        captured["encode_prompt"] = kwargs
        return {
            "qwen_prompt_embeds": torch.zeros(1, 2, 4),
            "qwen_attention_mask": torch.ones(1, 2),
            "t5_input_ids": torch.ones(1, 2, dtype=torch.long),
            "t5_attention_mask": torch.ones(1, 2),
            "negative_qwen_prompt_embeds": torch.zeros(1, 2, 4),
            "negative_qwen_attention_mask": torch.ones(1, 2),
            "negative_t5_input_ids": torch.ones(1, 2, dtype=torch.long),
            "negative_t5_attention_mask": torch.ones(1, 2),
        }

    def condition_prompt_embeds(**_kwargs):
        return torch.zeros(1, 2, 4)

    def prepare_latents(**kwargs):
        captured["prepare_latents"] = kwargs
        return torch.zeros(1, 16, 1, kwargs["height"] // 8, kwargs["width"] // 8)

    def diffuse(**kwargs):
        captured["diffuse"] = kwargs
        return kwargs["latents"]

    pipeline.encode_prompt = encode_prompt
    pipeline.condition_prompt_embeds = condition_prompt_embeds
    pipeline.prepare_latents = prepare_latents
    pipeline.prepare_timesteps = lambda **_kwargs: (torch.ones(1), 1)
    pipeline.diffuse = diffuse
    pipeline.decode_latents = lambda latents, output_type="pil": DiffusionOutput(output=latents)
    return pipeline, captured


def test_native_anima_forward_uses_official_default_resolution():
    pipeline, captured = _make_anima_forward_probe()
    req = OmniDiffusionRequest(
        prompts=["a red cube"],
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="anima-defaults",
    )

    pipeline.forward(req)

    assert captured["prepare_latents"]["height"] == 1024
    assert captured["prepare_latents"]["width"] == 1024


def test_native_anima_explicit_guidance_scale_drives_cfg_multiplier():
    pipeline, captured = _make_anima_forward_probe()
    req = OmniDiffusionRequest(
        prompts=["a red cube"],
        sampling_params=OmniDiffusionSamplingParams(guidance_scale=5.0),
        request_id="anima-guidance",
    )

    pipeline.forward(req)

    assert captured["diffuse"]["do_true_cfg"] is True
    assert captured["diffuse"]["true_cfg_scale"] == 5.0


def test_native_anima_true_cfg_scale_overrides_guidance_multiplier():
    pipeline, captured = _make_anima_forward_probe()
    req = OmniDiffusionRequest(
        prompts=["a red cube"],
        sampling_params=OmniDiffusionSamplingParams(guidance_scale=5.0, true_cfg_scale=3.0),
        request_id="anima-true-cfg",
    )

    pipeline.forward(req)

    assert captured["diffuse"]["do_true_cfg"] is True
    assert captured["diffuse"]["true_cfg_scale"] == 3.0


def test_enrich_config_single_file_rejects_unknown_pipeline(tmp_path):
    dummy_checkpoint = tmp_path / "model.safetensors"
    dummy_checkpoint.write_text("dummy")

    config = OmniDiffusionConfig(
        model=str(dummy_checkpoint),
        diffusion_load_format="diffusers_single_file",
        model_class_name="MissingPipeline",
    )
    with pytest.raises(ValueError, match="Could not find diffusers pipeline class MissingPipeline"):
        config.enrich_config()


def test_native_anima_cfg_equation():
    from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline

    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    pipeline.device = torch.device("cpu")
    pipeline._interrupt = False

    class MockProgressBar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def update(self):
            pass

    pipeline.progress_bar = lambda **kwargs: MockProgressBar()

    class MockTransformer:
        dtype = torch.float32

        def __call__(self, hidden_states, timestep, encoder_hidden_states, padding_mask, return_dict=False):
            # Return double the encoder_hidden_states to simulate transformer output
            return (encoder_hidden_states * 2.0,)

    pipeline.transformer = MockTransformer()

    class MockScheduler:
        def set_begin_index(self, index):
            pass

        def step(self, noise_pred, t, latents, return_dict=False):
            # Return the noise_pred itself to inspect it
            return (noise_pred,)

    pipeline.scheduler = MockScheduler()

    prompt_embeds = torch.tensor([[[2.0, 3.0]]])
    negative_prompt_embeds = torch.tensor([[[1.0, 2.0]]])
    latents = torch.zeros(1, 16, 1, 16, 16)
    padding_mask = torch.zeros(1, 1, 16, 16)
    timesteps = torch.tensor([500.0])

    # Cond = prompt_embeds * 2.0 = [4.0, 6.0]
    # Uncond = negative_prompt_embeds * 2.0 = [2.0, 4.0]
    # Expected noise_pred = uncond + true_cfg_scale * (cond - uncond)
    #                     = [2.0, 4.0] + 4.0 * ([4.0, 6.0] - [2.0, 4.0])
    #                     = [2.0, 4.0] + 4.0 * [2.0, 2.0]
    #                     = [2.0, 4.0] + [8.0, 8.0] = [10.0, 12.0]
    out_latents = pipeline.diffuse(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        latents=latents,
        padding_mask=padding_mask,
        timesteps=timesteps,
        do_true_cfg=True,
        true_cfg_scale=4.0,
    )

    expected = torch.tensor([[[10.0, 12.0]]])
    assert torch.allclose(out_latents, expected)
