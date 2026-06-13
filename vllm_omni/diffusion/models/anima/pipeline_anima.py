# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import logging
import os
from collections.abc import Iterable

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, Qwen3Model, T5TokenizerFast

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import DistributedAutoencoderKLQwenImage
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.anima.anima_text_conditioner import AnimaTextConditioner
from vllm_omni.diffusion.models.anima.anima_transformer import AnimaTransformer3DModel
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.size_utils import normalize_min_aligned_size

logger = logging.getLogger(__name__)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        if "timesteps" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        if "sigmas" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


_ANIMA_TRANSFORMER_CONFIG = {
    "in_channels": 16,
    "out_channels": 16,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 28,
    "mlp_ratio": 4.0,
    "text_embed_dim": 1024,
    "adaln_lora_dim": 256,
    "max_size": (128, 240, 240),
    "patch_size": (1, 2, 2),
    "rope_scale": (1.0, 4.0, 4.0),
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
}

_COSMOS_2_TRANSFORMER_RENAMES = {
    "t_embedder.1": "time_embed.t_embedder",
    "t_embedding_norm": "time_embed.norm",
    "blocks": "transformer_blocks",
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
    "self_attn": "attn1",
    "cross_attn": "attn2",
    "q_proj": "to_q",
    "k_proj": "to_k",
    "v_proj": "to_v",
    "output_proj": "to_out.0",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
    "x_embedder.proj.1": "patch_embed.proj",
    "final_layer.adaln_modulation.1": "norm_out.linear_1",
    "final_layer.adaln_modulation.2": "norm_out.linear_2",
    "final_layer.linear": "proj_out",
}

_COSMOS_2_TRANSFORMER_DROP_KEYS = (
    "accum_video_sample_counter",
    "accum_image_sample_counter",
    "accum_iteration",
    "accum_train_in_hours",
    "pos_embedder.seq",
    "pos_embedder.dim_spatial_range",
    "pos_embedder.dim_temporal_range",
    "_extra_state",
)


def get_anima_post_process_func(od_config: OmniDiffusionConfig):
    image_processor = VaeImageProcessor(vae_scale_factor=8)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


class AnimaPipeline(nn.Module, DiffusionPipelineProfilerMixin, ProgressBarMixin):
    """Native Anima model path.

    The Anima transformer checkpoint is distributed as a single safetensors
    file. This loader builds the native pipeline components and runs the denoise
    transformer/text-conditioner directly.
    """

    supports_step_execution = False

    def __init__(self, *, od_config: OmniDiffusionConfig, device=None):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = device or get_local_device()
        self.weights_sources = []

        self._raise_unsupported_features()

        self._interrupt = False
        self._guidance_scale = 1.0
        self._num_timesteps = 0
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _raise_unsupported_features(self):
        pc = self.od_config.parallel_config
        if pc.cfg_parallel_size > 1:
            raise NotImplementedError("AnimaPipeline does not yet support CFG parallel. Validate native CFG first.")
        if pc.sequence_parallel_size is not None and pc.sequence_parallel_size > 1:
            raise NotImplementedError("AnimaPipeline does not yet support sequence parallelism. Add an _sp_plan first.")
        if pc.tensor_parallel_size > 1:
            raise NotImplementedError("AnimaPipeline does not yet support tensor parallelism. Port TP linears first.")
        if pc.use_hsdp:
            raise NotImplementedError("AnimaPipeline does not yet support HSDP. Validate sharding after native load.")
        if self.od_config.cache_backend not in ("none", None):
            raise NotImplementedError("AnimaPipeline does not yet support TeaCache or Cache-DiT.")
        if self.od_config.quantization_config is not None:
            raise NotImplementedError("AnimaPipeline does not yet support quantized checkpoints.")
        if self.od_config.enable_cpu_offload:
            raise NotImplementedError("AnimaPipeline does not yet support CPU offload.")
        if self.od_config.enable_layerwise_offload:
            raise NotImplementedError("AnimaPipeline does not yet support layer-wise offload.")

    @staticmethod
    def _sub_state_dict(state_dict, prefix):
        return {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}

    @staticmethod
    def _infer_text_conditioner_config(state_dict):
        model_dim = state_dict["blocks.0.self_attn.q_proj.weight"].shape[0]
        source_dim = state_dict["blocks.0.cross_attn.k_proj.weight"].shape[1]
        target_vocab_size, target_dim = state_dict["embed.weight"].shape
        attention_head_dim = state_dict["blocks.0.self_attn.q_norm.weight"].shape[0]
        num_layers = 1 + max(int(key.split(".")[1]) for key in state_dict if key.startswith("blocks."))

        return {
            "source_dim": source_dim,
            "target_dim": target_dim,
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_attention_heads": model_dim // attention_head_dim,
            "target_vocab_size": target_vocab_size,
        }

    @staticmethod
    def _convert_cosmos_2_transformer_state_dict(state_dict):
        if "patch_embed.proj.weight" in state_dict:
            return state_dict

        converted_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.removeprefix("net.")
            if any(drop_key in new_key for drop_key in _COSMOS_2_TRANSFORMER_DROP_KEYS):
                continue
            for old_key, new_name in _COSMOS_2_TRANSFORMER_RENAMES.items():
                new_key = new_key.replace(old_key, new_name)
            if new_key in converted_state_dict:
                raise ValueError(f"Duplicate converted Anima transformer key: {new_key}")
            converted_state_dict[new_key] = value
        return converted_state_dict

    @staticmethod
    def _split_single_file_state_dict(state_dict):
        transformer_state_dict = AnimaPipeline._sub_state_dict(state_dict, "transformer.")
        text_conditioner_state_dict = AnimaPipeline._sub_state_dict(state_dict, "text_conditioner.")
        if transformer_state_dict and text_conditioner_state_dict:
            return transformer_state_dict, text_conditioner_state_dict

        text_conditioner_state_dict = AnimaPipeline._sub_state_dict(state_dict, "net.llm_adapter.")
        if text_conditioner_state_dict:
            transformer_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith("net.llm_adapter.")
            }
            return transformer_state_dict, text_conditioner_state_dict

        raise ValueError(
            "Anima native loader could not find text-conditioner weights. Expected either "
            "`text_conditioner.*` Diffusers keys or `net.llm_adapter.*` original Anima keys."
        )

    def _load_native_denoiser_components(self, state_dict=None):
        if state_dict is None:
            import os

            from safetensors.torch import load_file

            model_path = self.od_config.model
            if not os.path.isfile(model_path):
                raise ValueError(f"AnimaPipeline currently requires a local file path, got: {model_path}")

            state_dict = load_file(model_path, device="cpu")
        transformer_state_dict, text_conditioner_state_dict = self._split_single_file_state_dict(state_dict)
        del state_dict
        transformer_state_dict = self._convert_cosmos_2_transformer_state_dict(transformer_state_dict)

        native_text_conditioner = AnimaTextConditioner(
            **self._infer_text_conditioner_config(text_conditioner_state_dict)
        )
        native_text_conditioner.load_state_dict(text_conditioner_state_dict, strict=True)
        native_text_conditioner.to(device=self.device, dtype=self.od_config.dtype)
        del text_conditioner_state_dict

        native_transformer = AnimaTransformer3DModel(**_ANIMA_TRANSFORMER_CONFIG)
        native_transformer.load_state_dict(transformer_state_dict, strict=True)
        native_transformer.to(device=self.device, dtype=self.od_config.dtype)
        del transformer_state_dict

        return native_transformer, native_text_conditioner

    def _component_path(self, name, default_model):
        load_kwargs = self.od_config.diffusers_load_kwargs
        return load_kwargs.get(f"{name}_model", load_kwargs.get(f"{name}_path", default_model))

    @staticmethod
    def _from_pretrained_kwargs(model, subfolder=None):
        kwargs = {"local_files_only": os.path.exists(model)}
        if subfolder is not None and os.path.isdir(os.path.join(model, subfolder)):
            kwargs["subfolder"] = subfolder
        return kwargs

    def _load_outer_components(self):
        load_kwargs = self.od_config.diffusers_load_kwargs
        components_model = load_kwargs.get("components_model", load_kwargs.get("components_path"))
        if components_model is None:
            checkpoint_dir = os.path.dirname(os.path.abspath(self.od_config.model))
            components_model = checkpoint_dir if os.path.isdir(checkpoint_dir) else self.od_config.model

        text_encoder_model = self._component_path("text_encoder", components_model)
        vae_model = self._component_path("vae", components_model)
        tokenizer_model = self._component_path("tokenizer", components_model)
        t5_tokenizer_model = self._component_path("t5_tokenizer", components_model)

        text_encoder = Qwen3Model.from_pretrained(
            text_encoder_model,
            torch_dtype=self.od_config.dtype,
            **self._from_pretrained_kwargs(text_encoder_model, "text_encoder"),
        ).to(self.device)
        vae = DistributedAutoencoderKLQwenImage.from_pretrained(
            vae_model,
            torch_dtype=self.od_config.dtype,
            **self._from_pretrained_kwargs(vae_model, "vae"),
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            **self._from_pretrained_kwargs(tokenizer_model, "tokenizer"),
        )
        t5_tokenizer = T5TokenizerFast.from_pretrained(
            t5_tokenizer_model,
            **self._from_pretrained_kwargs(t5_tokenizer_model, "t5_tokenizer"),
        )

        scheduler_model = self._component_path("scheduler", components_model)
        has_scheduler_config = os.path.isfile(os.path.join(scheduler_model, "scheduler_config.json")) or os.path.isdir(
            os.path.join(scheduler_model, "scheduler")
        )
        if has_scheduler_config:
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                scheduler_model,
                **self._from_pretrained_kwargs(scheduler_model, "scheduler"),
            )
        else:
            scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

        return text_encoder, tokenizer, t5_tokenizer, vae, scheduler

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]] | None = None) -> set[str]:
        logger.info("Loading Anima transformer/text-conditioner directly from single-file checkpoint.")
        state_dict = dict(weights) if weights is not None else None
        loaded = set(state_dict.keys()) if state_dict is not None else set()
        if not state_dict:
            state_dict = None
        with set_current_diffusion_config(self.od_config):
            native_transformer, native_text_conditioner = self._load_native_denoiser_components(state_dict)
        text_encoder, tokenizer, t5_tokenizer, vae, scheduler = self._load_outer_components()

        self.transformer = native_transformer
        self.text_conditioner = native_text_conditioner
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.vae = vae
        self.scheduler = scheduler
        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if hasattr(self.vae, "temperal_downsample") else 8
        )
        return loaded

    def _extract_prompts(self, prompts):
        """Extract prompt and negative_prompt from OmniPromptType list."""
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts] or None
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in prompts):
            negative_prompt = None
        elif prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in prompts]
        else:
            negative_prompt = None
        return prompt, negative_prompt

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str],
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)
        if text_input_ids.shape[-1] == 0:
            text_input_ids = text_input_ids.new_zeros((text_input_ids.shape[0], 1))
            prompt_attention_mask = prompt_attention_mask.new_zeros((prompt_attention_mask.shape[0], 1))

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=False,
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = prompt_embeds * prompt_attention_mask.to(prompt_embeds).unsqueeze(-1)

        return prompt_embeds, prompt_attention_mask

    def _get_t5_prompt_ids(
        self,
        prompt: str | list[str],
        max_sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.t5_tokenizer(
            prompt,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device)

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        prepare_unconditional_embeds: bool = True,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor | None]:
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds, prompt_attention_mask = self._get_qwen_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        t5_input_ids, t5_attention_mask = self._get_t5_prompt_ids(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        negative_t5_input_ids = None
        negative_t5_attention_mask = None
        if prepare_unconditional_embeds:
            if negative_prompt is None:
                negative_prompt = ""
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            elif len(negative_prompt) != batch_size:
                raise ValueError(
                    f"`negative_prompt` length ({len(negative_prompt)}) must match prompt batch size ({batch_size})."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_qwen_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            negative_t5_input_ids, negative_t5_attention_mask = self._get_t5_prompt_ids(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        return {
            "qwen_prompt_embeds": prompt_embeds,
            "qwen_attention_mask": prompt_attention_mask,
            "t5_input_ids": t5_input_ids,
            "t5_attention_mask": t5_attention_mask,
            "negative_qwen_prompt_embeds": negative_prompt_embeds,
            "negative_qwen_attention_mask": negative_prompt_attention_mask,
            "negative_t5_input_ids": negative_t5_input_ids,
            "negative_t5_attention_mask": negative_t5_attention_mask,
        }

    def condition_prompt_embeds(
        self,
        qwen_prompt_embeds: torch.Tensor,
        qwen_attention_mask: torch.Tensor,
        t5_input_ids: torch.Tensor,
        t5_attention_mask: torch.Tensor,
        device: torch.device | None = None,
        conditioning_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        device = device or self.device
        conditioning_dtype = conditioning_dtype or self.text_conditioner.dtype
        output_dtype = output_dtype or self.transformer.dtype

        prompt_embeds = self.text_conditioner(
            source_hidden_states=qwen_prompt_embeds.to(device=device, dtype=conditioning_dtype),
            target_input_ids=t5_input_ids.to(device),
            target_attention_mask=t5_attention_mask.to(device),
            source_attention_mask=qwen_attention_mask.to(device),
        )
        return prompt_embeds.to(dtype=output_dtype, device=device)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = device or self.device
        dtype = dtype or self.transformer.dtype
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, 1, latent_height, latent_width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Draw latents in float32 then cast
        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        sigma_max = (
            getattr(self.scheduler.config, "sigma_max", 1.0)
            if hasattr(self, "scheduler") and hasattr(self.scheduler, "config")
            else 1.0
        )
        return latents.to(dtype=dtype) * sigma_max

    def prepare_timesteps(
        self,
        num_inference_steps: int,
        sigmas: list[float] | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, int]:
        device = device or self.device
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        return retrieve_timesteps(
            self.scheduler,
            device=device,
            sigmas=sigmas,
        )

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        latents: torch.Tensor,
        padding_mask: torch.Tensor,
        timesteps: torch.Tensor,
        do_true_cfg: bool,
        true_cfg_scale: float,
    ) -> torch.Tensor:
        self.scheduler.set_begin_index(0)

        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t

                # Broadcast timestep to match batch size
                timestep = t.expand(latents.shape[0]).to(dtype=self.transformer.dtype, device=latents.device)
                timestep = timestep / 1000.0  # normalize to 0..1 range

                # Predict noise
                latent_model_input = latents.to(self.transformer.dtype)

                # Positive pass
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]

                # Negative / uncond pass
                if do_true_cfg and negative_prompt_embeds is not None:
                    noise_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + true_cfg_scale * (noise_pred - noise_pred_uncond)

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                pbar.update()

        return latents

    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: str = "pil",
    ) -> DiffusionOutput:
        """Decode final latents."""
        if output_type == "latent":
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str | None = "pil",
        max_sequence_length: int = 512,
    ) -> DiffusionOutput:
        extracted_prompt, negative_prompt = self._extract_prompts(req.prompts)
        prompt = extracted_prompt or prompt

        height = req.sampling_params.height or height or 1024
        width = req.sampling_params.width or width or 1024
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)

        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        sigmas = req.sampling_params.sigmas or sigmas
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length
        generator = req.sampling_params.generator or generator
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
            if req.sampling_params.true_cfg_scale is None:
                true_cfg_scale = guidance_scale
        if req.sampling_params.true_cfg_scale is not None:
            true_cfg_scale = req.sampling_params.true_cfg_scale
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        do_true_cfg = guidance_scale > 1.0

        # Encode prompts
        enc_out = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prepare_unconditional_embeds=do_true_cfg,
            max_sequence_length=max_sequence_length,
            device=self.device,
            dtype=self.text_encoder.dtype,
        )

        # Condition prompt embeds
        prompt_embeds = self.condition_prompt_embeds(
            qwen_prompt_embeds=enc_out["qwen_prompt_embeds"],
            qwen_attention_mask=enc_out["qwen_attention_mask"],
            t5_input_ids=enc_out["t5_input_ids"],
            t5_attention_mask=enc_out["t5_attention_mask"],
            device=self.device,
        )

        negative_prompt_embeds = None
        if do_true_cfg and enc_out["negative_qwen_prompt_embeds"] is not None:
            negative_prompt_embeds = self.condition_prompt_embeds(
                qwen_prompt_embeds=enc_out["negative_qwen_prompt_embeds"],
                qwen_attention_mask=enc_out["negative_qwen_attention_mask"],
                t5_input_ids=enc_out["negative_t5_input_ids"],
                t5_attention_mask=enc_out["negative_t5_attention_mask"],
                device=self.device,
            )

        # Repeat for num_images_per_prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if negative_prompt_embeds is not None:
            _, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # Prepare latents
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=16,  # Cosmos/Anima has 16 channels
            height=height,
            width=width,
            dtype=self.transformer.dtype,
            device=self.device,
            generator=generator,
            latents=latents,
        )

        # Prepare padding mask
        padding_mask = latents.new_zeros(1, 1, height, width, dtype=self.transformer.dtype)

        # Prepare timesteps
        timesteps, num_inference_steps = self.prepare_timesteps(
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            device=self.device,
        )
        self._num_timesteps = len(timesteps)

        # Denoise loop
        latents = self.diffuse(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            latents=latents,
            padding_mask=padding_mask,
            timesteps=timesteps,
            do_true_cfg=do_true_cfg,
            true_cfg_scale=true_cfg_scale,
        )

        self._current_timestep = None
        return self.decode_latents(latents, output_type=output_type)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt
