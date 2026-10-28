"""P1 production-manager falsification screen. No encoder, VAE, or HTTP run.

Load the actual DiT and saved context; native prefill generates both branches.
The unmodified native manager and P1 manager share immutable request prefixes,
but own independent static buffers. Two target inputs come from native Euler
steps at the saved 40-step schedule. Stop on correctness/capture failure.
Inspect emitted copy behavior before authorizing any timing follow-up.
"""

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import traceback
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import torch


def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def storage_bytes(entries):
    storages = {}
    for entry in entries:
        for tensor in [*entry.k, *entry.v, *(entry.packed_k or []), *(entry.packed_v or [])]:
            s = tensor.untyped_storage()
            storages[s.data_ptr()] = s.nbytes()
    return sum(storages.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--context', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extend', action='store_true')
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

        registrations = []

        def register(manager, request_cache):
            for metadata in registrations:
                manager.register_prefill(kv_cache=request_cache, **metadata)

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
            register(candidate, cache)
            entries = list(candidate.entries.values())
            check('p1_all_layer_buffers', len(entries) == 2 and all(len(e.packed_k or []) == 32 for e in entries))
            result['memory'] = {'native_graph_kv_bytes': storage_bytes(native.entries.values()),
                'p1_graph_kv_bytes': storage_bytes(entries), 'allocated_before_capture': torch.cuda.memory_allocated(),
                'reserved_before_capture': torch.cuda.memory_reserved()}
            pointers = [[t.data_ptr() for t in [*e.packed_k, *e.packed_v]] for e in entries]
            check('branch_distinct_storage', len(set(sum(pointers, []))) == 128)
            latest_args = {}
            decode_inputs = []
            tails = {}
            for step in (1, 2):
                preds = {}
                for branch in ('cond', 'uncond'):
                    kw = call_args(branch, latents, times[step])
                    latest_args[branch] = kw
                    decode_inputs.append((step, branch, kw))
                    baseline = native.try_decode(**kw)
                    check(f's{step}_{branch}_native_graph', baseline is not None)
                    aa = native.try_decode(**kw)
                    check(f's{step}_{branch}_native_AA', bits(baseline, aa))
                    actual = candidate.try_decode(**kw)
                    check(f's{step}_{branch}_p1_graph', actual is not None)
                    check(f's{step}_{branch}_output_exact', bits(baseline, actual),
                        max_abs=float((baseline.float() - actual.float()).abs().max()))
                    e = next(e for e in entries if e.branch == branch)
                    check(f's{step}_{branch}_prefix_unchanged', all(
                        bits(e.k[i], c[branch]['key']) and bits(e.v[i], c[branch]['value']) for i, c in enumerate(cache)))
                    # Observe every layer's target; these snapshots are outside graph replay.
                    if step == 2:
                        check(f's{step}_{branch}_all_targets_changed', all(
                            not bits(old, now[:, e.prefix_len:]) for old, now in zip(tails[branch], [*e.packed_k, *e.packed_v])))
                    tails[branch] = [t[:, e.prefix_len:].clone() for t in [*e.packed_k, *e.packed_v]]
                    check(f's{step}_{branch}_single_capture', e.captures == 1 and e.graph is not None)
                    preds[branch] = baseline
                if step == 1:
                    latents = advance(preds, times[step], latents)
            check('stable_all_layer_pointers', pointers == [[t.data_ptr() for t in [*e.packed_k, *e.packed_v]] for e in entries])
            check('request_cache_has_no_scratch', all(set(c[b]) == {'key', 'value'} for c in cache for b in ('cond', 'uncond')))
            expected_last = {b: preds[b].clone() for b in ('cond', 'uncond')}
            del tails, preds, baseline, actual, aa
            gc.collect()
            torch.cuda.synchronize()
            result['memory'].update(allocated_after_capture=torch.cuda.memory_allocated(), reserved_after_capture=torch.cuda.memory_reserved())

            if a.extend:
                from extend import run
                run(model, native, candidate, decode_inputs, check, bits, result, save)

            # Separate untimed traces establish dispatched copies. Raw traces stay remote.
            for name, manager in [('B0', native), ('P1', candidate)]:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                    for branch in ('cond', 'uncond'):
                        manager.try_decode(**latest_args[branch])
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(a.output / (name + '-replay.json')))
            print('PROFILES saved; checking live owner conflict', flush=True)

            # Only tensor0 needs a new identity to represent the second owner;
            # all other immutable prefixes may alias without being written.
            second = QwenImage21KVCache([{b: dict(v) for b, v in c.items()} for c in cache])
            for b in ('cond', 'uncond'):
                second[0][b]['key'] = second[0][b]['key'].clone()
            register(candidate, second)
            for b in ('cond', 'uncond'):
                conflict = dict(latest_args[b], kv_cache=second)
                check(b + '_live_owner_rejected', candidate.try_decode(**conflict) is None)
            check('conflict_preserves_prefix', all(bits(e.k[i], cache[i][e.branch]['key']) for e in entries for i in range(32)))
            candidate.release_idle_entries()
            check('live_entries_preserved', len(candidate.entries) == 2)

            # Drop diagnostic references to entries and request0 ownership.
            entry_refs = [weakref.ref(e) for e in entries]
            buffer_refs = [weakref.ref(t) for e in entries for t in [*e.packed_k, *e.packed_v]]
            del entries, e
            cache.clear()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            candidate.release_idle_entries()
            native.release_idle_entries()
            gc.collect()
            check('idle_entries_released', not candidate.entries and not candidate._owners and not native.entries)
            check('entry_and_full_buffers_released', all(ref() is None for ref in entry_refs + buffer_refs))
            result['memory'].update(allocated_before_release=before, allocated_after_release=torch.cuda.memory_allocated(),
                reserved_after_release=torch.cuda.memory_reserved())
            register(candidate, second)
            for b in ('cond', 'uncond'):
                changed_owner = dict(latest_args[b], kv_cache=second)
                out = candidate.try_decode(**changed_owner)
                check(b + '_new_owner_recapture_exact', out is not None and bits(out, expected_last[b]))
            second.clear()
            candidate.release_idle_entries()
            check('second_request_cleanup', not candidate.entries and not candidate._owners)
            result['second_request'] = 'Same-prefix contents, distinct owner, conflict then recapture and idle release; not a new-prompt numerical test.'
            result['attention_operands'] = 'See attention_observation' if a.extend else 'Not yet observed individually; complete graph output checks are separate.'
            result['status'] = 'copy_inspection_pending'
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
    return 0 if result['status'] == 'copy_inspection_pending' else 2


if __name__ == '__main__':
    raise SystemExit(main())
