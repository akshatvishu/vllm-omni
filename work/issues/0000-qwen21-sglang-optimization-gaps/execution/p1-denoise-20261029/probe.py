"""Measure native/P1 production 40-step denoising on the saved four-reference context."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import torch


def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--context', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'checks': [], 'scope': 'Production 32-layer decode managers; no image/request qualification'}

    def save():
        (a.output / 'findings.json').write_text(json.dumps(result, indent=2) + '\n')

    def check(name, ok, **detail):
        row = {'name': name, 'passed': bool(ok), **detail}
        result['checks'].append(row)
        print(json.dumps(row), flush=True)
        save()
        if not ok:
            raise RuntimeError('gate failed: ' + name)

    initialized = False
    try:
        from safetensors import safe_open
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm_omni.diffusion.compile import regionally_compile
        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
        from vllm_omni.diffusion.distributed.parallel_state import (
            init_distributed_environment, initialize_model_parallel, destroy_distributed_env,
        )
        from vllm_omni.diffusion.forward_context import set_forward_context
        from vllm_omni.diffusion.models.qwen_image_21 import decode_graph as dg
        from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
        from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
            QwenImage21KVCache, QwenImage21Transformer2DModel,
        )

        torch.cuda.set_device(0)
        prop = torch.cuda.get_device_properties(0)
        check('gfx942_stack', 'gfx942' in prop.gcnArchName and torch.version.hip is not None,
              device=prop.name, arch=prop.gcnArchName)
        result['stack'] = {'torch': torch.__version__, 'torch_git': torch.version.git_version, 'hip': torch.version.hip}
        result['source_sha'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        result['context_sha256'] = hashlib.sha256(a.context.read_bytes()).hexdigest()
        check('frozen_context', result['context_sha256'] == 'a26ac7fcca255299c23adac5f193c6512832a326e916815db042628f9dfe3500')
        ctx_cpu = torch.load(a.context, map_location='cpu', weights_only=True)
        ctx = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in ctx_cpu.items()}
        check('four_reference_cfg', len(ctx['img_shapes'][0]) == 5 and ctx['do_true_cfg'])
        vcfg = VllmConfig()
        od = OmniDiffusionConfig(model=str(a.checkpoint), dtype=torch.bfloat16,
            diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend='TORCH_SDPA')))
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
            distributed_init_method='file://' + str(a.output / 'rendezvous'))
        initialized = True
        initialize_model_parallel(data_parallel_size=1, cfg_parallel_size=1, sequence_parallel_size=1,
            ulysses_degree=1, ring_degree=1, tensor_parallel_size=1, pipeline_parallel_size=1)
        config = json.loads((a.checkpoint / 'transformer/config.json').read_text())
        kwargs = {k: v for k, v in config.items() if k in inspect.signature(QwenImage21Transformer2DModel).parameters}
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with set_current_vllm_config(vcfg), set_current_diffusion_config(od), torch.device('cuda'):
                model = QwenImage21Transformer2DModel(od_config=od, **kwargs).eval()
        finally:
            torch.set_default_dtype(old_dtype)

        def weights():
            for shard in sorted((a.checkpoint / 'transformer').glob('*.safetensors')):
                with safe_open(shard, framework='pt', device='cpu') as reader:
                    for key in reader.keys():
                        yield key, reader.get_tensor(key)

        with torch.no_grad():
            loaded = model.load_weights(weights())
        check('all_weights_loaded', not (set(dict(model.named_parameters())) - loaded))
        check('native_bf16_32_blocks', len(model.transformer_blocks) == 32 and all(p.dtype == torch.bfloat16 for p in model.parameters()))
        check('sdpa_backend', model.transformer_blocks[0].attn.attn.attn_backend.get_name() == 'SDPA')
        regionally_compile(model, dynamic=od.diffusion_compile_dynamic,
            options={'triton.cudagraphs': False, 'triton.cudagraph_trees': False})
        check('regional_compile', all(hasattr(b.forward, '_torchdynamo_orig_callable') for b in model.transformer_blocks),
              dynamic=od.diffusion_compile_dynamic)

        with set_forward_context(vllm_config=vcfg, omni_diffusion_config=od), torch.inference_mode():
            from denoise import run
            run(model, ctx, a.checkpoint, a.output, check, result, save)
            result['status'] = 'completed'
            save()
    except BaseException:
        result['status'] = 'failed'
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        if initialized:
            try:
                destroy_distributed_env()
            except Exception:
                result['cleanup_error'] = traceback.format_exc()
        save()
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
