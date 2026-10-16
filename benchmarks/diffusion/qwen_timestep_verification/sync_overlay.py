# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental import overlay. Production checkout remains untouched."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys

PREFIX = "vllm_omni.diffusion.models.qwen_image."
SOURCES = {"cfg_parallel": "", "pipeline_qwen_image": ""}


def replace_once(source, before, after):
    assert source.count(before) == 1, (before, source.count(before))
    return source.replace(before, after)


def transform(name, source):
    if name == "cfg_parallel":
        source = replace_once(source,
            "        with self.progress_bar(total=len(timesteps)) as pbar:",
            "        model_timesteps = timesteps.to(device=latents.device, dtype=latents.dtype)\n\n"
            "        with self.progress_bar(total=len(timesteps)) as pbar:")
        return replace_once(source,
            "timestep = t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)",
            "timestep = model_timesteps[i].expand(latents.shape[0])")
    source = replace_once(source,
        '        state.timesteps = ctx["timesteps"]',
        '        state.timesteps = ctx["timesteps"]\n'
        '        state.extra["sync_model_timesteps"] = state.timesteps.to(\n'
        '            device=state.latents.device, dtype=state.latents.dtype\n'
        '        )')
    return replace_once(source,
        "            timestep=t,\n",
        '            timestep=torch.cat([\n'
        '                state.extra["sync_model_timesteps"][state.step_index].expand(state.latents.shape[0])\n'
        '                for state in input_batch.states\n'
        '            ]),\n')


def instrument(module):
    """Correctness audit only; never enabled for request timing."""
    import torch
    cls = module.QwenImagePipeline
    original_predict = cls.predict_noise_maybe_with_cfg
    original_scheduler = cls.scheduler_step_maybe_with_cfg
    original_decode = cls._decode_latents
    original_build = cls._build_denoise_kwargs

    def predict(self, do_true_cfg, true_cfg_scale, positive_kwargs, *args, **kwargs):
        records = self.__dict__.setdefault("_sync_model_records", [])
        t = positive_kwargs["timestep"]
        records.append((str(t.device), str(t.dtype), t.detach().cpu()))
        return original_predict(self, do_true_cfg, true_cfg_scale, positive_kwargs, *args, **kwargs)

    def scheduler(self, noise_pred, t, *args, **kwargs):
        records = self.__dict__.setdefault("_sync_scheduler_records", [])
        records.append((str(t.device), str(t.dtype), t.detach().cpu()))
        assert t.device.type == "cpu", "Experiment must preserve CPU scheduler timestep"
        return original_scheduler(self, noise_pred, t, *args, **kwargs)

    def build(self, *args, **kwargs):
        self.__dict__.setdefault("_sync_build_input_devices", []).append(str(kwargs["timestep"].device))
        return original_build(self, *args, **kwargs)

    def decode(self, latents, *args, **kwargs):
        index = self.__dict__.get("_sync_decode_index", 0)
        self._sync_decode_index = index + 1
        output = Path(os.environ["QWEN_SYNC_AUDIT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        torch.save({
            "latents": latents.detach().cpu(),
            "model_timesteps": self.__dict__.pop("_sync_model_records", []),
            "scheduler_timesteps": self.__dict__.pop("_sync_scheduler_records", []),
            "build_input_devices": self.__dict__.pop("_sync_build_input_devices", []),
        }, output / f"request-{index:03d}.pt")
        return original_decode(self, latents, *args, **kwargs)

    cls.predict_noise_maybe_with_cfg = predict
    cls.scheduler_step_maybe_with_cfg = scheduler
    cls._build_denoise_kwargs = build
    cls._decode_latents = decode


class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path).decode("utf-8")
        name = fullname.rsplit(".", 1)[-1]
        manifest = json.loads((Path(__file__).parent / "source-manifest.json").read_text())
        assert hashlib.sha256(source.encode()).hexdigest() == manifest[name], self.path
        if os.environ["QWEN_SYNC_ARM"] == "timestep":
            source = transform(name, source)
        return compile(source, self.path, "exec")

    def exec_module(self, module):
        super().exec_module(module)
        if module.__name__.endswith(".pipeline_qwen_image"):
            if os.environ.get("QWEN_SYNC_AUDIT_DIR"):
                instrument(module)
            proof = Path(os.environ["QWEN_SYNC_OUTPUT"]) / f"overlay-{os.getpid()}.json"
            proof.write_text(json.dumps({"arm": os.environ["QWEN_SYNC_ARM"], "module": module.__file__}))


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in {PREFIX + name for name in SOURCES}:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec and spec.origin
        spec.loader = Loader(fullname, spec.origin)
        return spec


def install():
    if os.environ.get("QWEN_SYNC_ARM") and not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
