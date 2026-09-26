# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Check that C0 pipelines set TeaCache's CFG flag before dispatching a step."""

from collections.abc import Callable
from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StopAtDenoiserError(Exception):
    pass


class _PipelineStub:
    vae_scale_factor = 8
    default_sample_size = 4
    is_distilled = False
    _execution_device = torch.device("cpu")
    device = torch.device("cpu")

    def __init__(self, model, pipeline_class):
        self.model = model
        self.pipeline_class = pipeline_class
        self._guidance_scale = 1.0
        self._attention_kwargs = None
        self._interrupt = False
        self.predict_noise_maybe_with_cfg: Callable[..., Any]
        self.transformer = SimpleNamespace(
            config=SimpleNamespace(in_channels=16), guidance_embeds=None, dtype=torch.float32
        )
        self.scheduler = SimpleNamespace(config={}, set_begin_index=lambda index: None)
        self.check_inputs = lambda *args, **kwargs: None
        self.check_cfg_parallel_validity = lambda *args, **kwargs: None
        self.encode_prompt = lambda **kwargs: (torch.ones(1, 2, 2), torch.zeros(2, 3))
        self.rewire_prompt = lambda prompt, device: prompt
        self.cfg_normalize_function = lambda *args, **kwargs: None
        self.progress_bar = lambda **kwargs: nullcontext(SimpleNamespace(update=lambda: None))
        self.prepare_latents = self._prepare_edit_latents if model == "longcat_edit" else self._prepare_latents

    @staticmethod
    def _prepare_latents(*args, **kwargs):
        return torch.ones(1, 4, 2), torch.zeros(4, 3)

    @staticmethod
    def _prepare_edit_latents(*args, **kwargs):
        return torch.ones(1, 4, 2), torch.ones(1, 2, 2), torch.zeros(4, 3), torch.zeros(2, 3)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self.pipeline_class.do_classifier_free_guidance.fget(self)

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt


_PIPELINES = [
    ("flux2", "vllm_omni.diffusion.models.flux2.pipeline_flux2", "Flux2Pipeline", False),
    ("klein", "vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein", "Flux2KleinPipeline", False),
    ("klein", "vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein", "Flux2KleinPipeline", True),
    ("longcat", "vllm_omni.diffusion.models.longcat_image.pipeline_longcat_image", "LongCatImagePipeline", False),
    (
        "longcat_edit",
        "vllm_omni.diffusion.models.longcat_image.pipeline_longcat_image_edit",
        "LongCatImageEditPipeline",
        False,
    ),
]


@pytest.mark.parametrize(("model", "module_name", "class_name", "distilled"), _PIPELINES)
def test_pipeline_cfg_flag_precedes_denoiser_and_clears_between_requests(
    monkeypatch, model, module_name, class_name, distilled
):
    module = import_module(module_name)
    monkeypatch.setattr(module, "retrieve_timesteps", lambda *args, **kwargs: (torch.tensor([1000.0]), 1))
    pipeline_class = getattr(module, class_name)
    pipeline = _PipelineStub(model, pipeline_class)
    pipeline.is_distilled = distilled
    forward = pipeline_class.forward

    for enabled in (True, False):
        expected_cfg = enabled and not distilled
        prompt = {
            "prompt": "image",
            "negative_prompt": "bad" if enabled else None,
            "additional_information": {
                "prompt_image": torch.zeros(3, 32, 32),
                "preprocessed_image": torch.zeros(3, 32, 32),
            },
        }
        sampling_params = SimpleNamespace(
            height=32,
            width=32,
            num_inference_steps=1,
            sigmas=None,
            guidance_scale=2.0 if enabled else 1.0,
            generator=None,
            num_outputs_per_prompt=1,
            max_sequence_length=64,
            extra_args={"enable_prompt_rewrite": False},
            latents=None,
            strength=None,
            output_type="latent",
        )
        req = SimpleNamespace(prompts=[prompt], sampling_params=sampling_params)

        def stop_at_denoiser(**kwargs):
            assert pipeline.transformer.do_true_cfg is expected_cfg
            assert kwargs["do_true_cfg"] is expected_cfg
            assert (kwargs["negative_kwargs"] is not None) is expected_cfg
            raise _StopAtDenoiserError

        pipeline.predict_noise_maybe_with_cfg = stop_at_denoiser
        with pytest.raises(_StopAtDenoiserError):
            forward(pipeline, req)
