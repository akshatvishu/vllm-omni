"""Verify compiled decode fallback repair and capture-failure recovery.

Reuse actual checkpoint, saved four-reference context and the first native Euler
step. Preserve P1 source, compile policy and arithmetic. Require output reproduction before attributing boundary differences or testing argument swaps.
No timing, profiling, encoder, VAE or full requests.
"""

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

        cpu_tests = subprocess.run([
            'python', '-m', 'pytest', '-q',
            'tests/diffusion/models/qwen_image_21/test_shared_modulation.py',
            'tests/diffusion/models/qwen_image_21/test_decode_graph.py', '-m', 'cpu',
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (a.output / 'cpu-tests.log').write_text(cpu_tests.stdout)
        check('existing_cpu_tests', cpu_tests.returncode == 0, summary=cpu_tests.stdout[-2000:])
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

        native = model._decode_graph_manager
        candidate = dg.QwenImage21DecodeGraphManager(model)
        entry_cls = dg.QwenImage21DecodeGraphEntry

        def native_entry(**kw):
            kw['retain_attention_buffers'] = False
            return entry_cls(**kw)

        cache = QwenImage21KVCache([{} for _ in model.transformer_blocks])
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(a.checkpoint), subfolder='scheduler', local_files_only=True)
        holder = SimpleNamespace(scheduler=scheduler, device=torch.device('cuda'))
        times, _ = QwenImage21Pipeline.prepare_timesteps(holder, 40, None, ctx['latents'].shape[1])
        check('saved_schedule_exact', bits(times, ctx['timesteps']))
        scheduler.set_begin_index(0)
        latents = ctx['latents'].clone()
        s = latents.shape[1]

        def call_args(branch, lat, time, request_cache=cache):
            negative = branch == 'uncond'
            return dict(hidden_states=torch.cat([ctx['image_latents'], lat], dim=1),
                timestep=time.expand(1).to(lat.dtype) / 1000,
                kv_cache=request_cache, cache_branch=branch, img_shapes=ctx['img_shapes'][0],
                img_mask=ctx['negative_img_mask' if negative else 'img_mask'],
                encoder_hidden_states_mask=ctx['negative_prompt_embeds_mask' if negative else 'prompt_embeds_mask'])

        def advance(preds, time, lat):
            combined = preds['uncond'] + 4.0 * (preds['cond'] - preds['uncond'])
            return scheduler.step(combined, time, lat, return_dict=False)[0]

        with set_forward_context(vllm_config=vcfg, omni_diffusion_config=od), torch.inference_mode():
            model.do_true_cfg = True
            print('PREFILL starting both CFG branches', flush=True)
            preds = {}
            with patch.object(dg, 'QwenImage21DecodeGraphEntry', native_entry):
                for branch in ('cond', 'uncond'):
                    kw = call_args(branch, latents, times[0])
                    kw['img_shapes'] = ctx['img_shapes']
                    kw['encoder_hidden_states'] = ctx['negative_prompt_embeds' if branch == 'uncond' else 'prompt_embeds']
                    preds[branch] = model(**kw, return_dict=False)[0][:, -s:].clone()
            check('native_prefill_entries', len(native.entries) == 2 and all(e.packed_k is None for e in native.entries.values()))
            latents = advance(preds, times[0], latents)
            del preds
            registrations = [dict(cache_branch=e.branch, prefix_len=e.prefix_len,
                img_shapes=ctx['img_shapes'][0], target_freqs=e.freqs.clone(), joint_key_valid=None, dtype=latents.dtype)
                for e in native.entries.values()]
            # Reuse prefill and its first Euler update, but do not repeat lifecycle checks.
            native.clear()
            latest_args = {b: call_args(b, latents, times[1]) for b in ('cond', 'uncond')}
            from recovery import run
            run(model, native, candidate, cache, registrations, latest_args,
                ctx, native_entry, check, result, save, a.output)
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
