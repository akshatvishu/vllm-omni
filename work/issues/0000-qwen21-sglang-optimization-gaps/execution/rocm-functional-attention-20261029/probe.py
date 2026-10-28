# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate a functional AITER boundary in one complete compiled block."""

import argparse
import copy
import os
import types
import functools
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import traceback
from unittest.mock import patch

import torch


def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@functools.wraps(torch.nn.functional.scaled_dot_product_attention)
def forbid_sdpa(*args, **kwargs):
    raise AssertionError('SDPA is forbidden in this experiment')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--context', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', checks=[], cases=[], scope='one-block functional AITER integration; no timing')

    def save():
        (a.output / 'findings.json').write_text(json.dumps(result, indent=2) + '\n')

    def check(name, ok, **detail):
        row = dict(name=name, passed=bool(ok), **detail)
        result['checks'].append(row)
        print(json.dumps(row), flush=True)
        save()
        if not ok:
            raise RuntimeError('gate failed: ' + name)

    initialized = False
    try:
        from safetensors import safe_open
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm_omni.diffusion.attention.backends.utils import fa
        from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import build_segments
        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
        from vllm_omni.diffusion.distributed.parallel_state import (
            init_distributed_environment, initialize_model_parallel, destroy_distributed_env,
        )
        from vllm_omni.diffusion.forward_context import set_forward_context
        from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
            QwenImage21KVCache, QwenImage21Transformer2DModel,
        )
        import aiter.ops.mha as mha
        import candidate

        torch.cuda.set_device(0)
        prop = torch.cuda.get_device_properties(0)
        check('gfx942', torch.version.hip is not None and 'gfx942' in prop.gcnArchName,
              device=prop.name, arch=prop.gcnArchName)
        check('aiter_binding', fa.IS_AITER and fa.flash_attn_func is mha.flash_attn_func)
        check('gpu_health_before', torch.ones(128, device='cuda').sum().item() == 128)
        result['stack'] = dict(torch=torch.__version__, torch_git=torch.version.git_version,
                               hip=torch.version.hip, aiter_file=inspect.getfile(mha),
                               aiter_sha256=hashlib.sha256(Path(inspect.getfile(mha)).read_bytes()).hexdigest())
        check('pinned_runtime', torch.version.git_version == '733fca1901cfa5094481cfa9dfc8f8483e18f491'
              and result['stack']['aiter_sha256'] == '7e95da87135af31ab49fdc09f7fc84d7f8d780e7ffa33a7143d3d4b9a669f503')
        result['sha'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        result['context_sha256'] = hashlib.sha256(a.context.read_bytes()).hexdigest()
        check('saved_context', result['context_sha256'] == 'a26ac7fcca255299c23adac5f193c6512832a326e916815db042628f9dfe3500')
        ctx_cpu = torch.load(a.context, map_location='cpu', weights_only=True)
        ctx = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in ctx_cpu.items()}
        vcfg = VllmConfig()
        od = OmniDiffusionConfig(model=str(a.checkpoint), dtype=torch.bfloat16,
            diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend='FLASH_ATTN')))
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
            distributed_init_method='file://' + str(a.output / 'rendezvous'))
        initialized = True
        initialize_model_parallel(data_parallel_size=1, cfg_parallel_size=1, sequence_parallel_size=1,
            ulysses_degree=1, ring_degree=1, tensor_parallel_size=1, pipeline_parallel_size=1)
        config = json.loads((a.checkpoint / 'transformer/config.json').read_text())
        kwargs = {k: v for k, v in config.items() if k in inspect.signature(QwenImage21Transformer2DModel).parameters}
        kwargs['num_layers'] = 1
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
                        name = key.removeprefix('transformer.')
                        if name.startswith('transformer_blocks.') and not name.startswith('transformer_blocks.0.'):
                            continue
                        yield key, reader.get_tensor(key)

        with torch.no_grad():
            loaded = model.load_weights(weights())
        check('all_used_weights_loaded', not (set(dict(model.named_parameters())) - loaded))
        attention = model.transformer_blocks[0].attn.attn
        check('rocm_flash_backend', attention.attn_backend.get_name() == 'FLASH_ATTN',
              backend=attention.attn_backend.get_name())

        def equal_tree(x, y):
            if isinstance(x, torch.Tensor):
                return isinstance(y, torch.Tensor) and bits(x, y)
            if isinstance(x, dict):
                return x.keys() == y.keys() and all(equal_tree(x[k], y[k]) for k in x)
            if isinstance(x, (tuple, list)):
                return len(x) == len(y) and all(equal_tree(a, b) for a, b in zip(x, y, strict=True))
            return x == y

        def clone_tree(x):
            if isinstance(x, torch.Tensor):
                return x.clone()
            if isinstance(x, dict):
                return {k: clone_tree(v) for k, v in x.items()}
            if isinstance(x, tuple):
                return tuple(clone_tree(v) for v in x)
            return x

        class Captured(Exception):
            pass

        block = model.transformer_blocks[0]
        reference_block = copy.deepcopy(block)
        reference_block.attn.attn.forward = types.MethodType(candidate.reference_forward, reference_block.attn.attn)
        original_flash = fa.flash_attn_func
        options = {'triton.cudagraphs': False, 'triton.cudagraph_trees': False}
        from torch._inductor import config as inductor_config
        check('default_v2_enabled', inductor_config.enable_auto_functionalized_v2 is True)
        result['compile_options'] = options
        compiled = {name: torch.compile(module.forward, dynamic=od.diffusion_compile_dynamic,
                                        fullgraph=True, options=options)
                    for name, module in (('reference', reference_block), ('candidate', block))}
        check('functional_schema', all(not arg.alias_info or not arg.alias_info.is_write
              for arg in candidate.segment._opoverload._schema.arguments))

        def capture(branch, cache, step):
            captured = []

            def hook(module, positional, keywords):
                assert not positional
                captured.append(dict(keywords))
                raise Captured()

            handle = block.register_forward_pre_hook(hook, with_kwargs=True)
            negative = branch == 'uncond'
            try:
                model(hidden_states=torch.cat([ctx['image_latents'], ctx['latents']], dim=1),
                    encoder_hidden_states=ctx['negative_prompt_embeds' if negative else 'prompt_embeds'],
                    timestep=ctx['timesteps'][step].expand(1).to(ctx['latents'].dtype) / 1000,
                    img_shapes=ctx['img_shapes'],
                    img_mask=ctx['negative_img_mask' if negative else 'img_mask'],
                    encoder_hidden_states_mask=ctx['negative_prompt_embeds_mask' if negative else 'prompt_embeds_mask'],
                    kv_cache=cache, cache_branch=branch, return_dict=False)
            except Captured:
                pass
            finally:
                handle.remove()
            assert len(captured) == 1
            return captured[0]

        def run(fn, kw):
            local = dict(kw)
            local['kv_cache'] = clone_tree(kw['kv_cache'])
            out = fn(**local)
            return out, local['kv_cache']

        with set_forward_context(vllm_config=vcfg, omni_diffusion_config=od), torch.inference_mode(), patch(
            'torch.nn.functional.scaled_dot_product_attention', new=forbid_sdpa,
        ), patch.object(fa, 'flash_attn_func', candidate.functional_flash):
            model.do_true_cfg = True
            for branch in ('cond', 'uncond'):
                cache = QwenImage21KVCache([{}])
                for phase in ('prefill', 'decode'):
                    label = branch + '_' + phase
                    kw = capture(branch, cache, 0 if phase == 'prefill' else 1)
                    n = kw['hidden_states'].shape[1]
                    check(label + '_geometry', n == (19383 if branch == 'cond' else 19367)
                          if phase == 'prefill' else n == 3072, tokens=n,
                          cache_write_len=kw['cache_write_len'])
                    saved_inputs = {k: clone_tree(v) for k, v in kw.items()
                                    if isinstance(v, (torch.Tensor, tuple, dict))}
                    case = dict(branch=branch, phase=phase, tokens=n)
                    result['cases'].append(case)
                    with patch.object(fa, 'flash_attn_func', original_flash):
                        eager_native = run(block.forward, kw)
                    eager_candidate = run(block.forward, kw)
                    check(label + '_eager_exact', equal_tree(eager_native, eager_candidate))
                    del eager_native, eager_candidate

                    candidate.AUDIT = {'segments': []}
                    expected = run(compiled['reference'], kw)
                    actual = run(compiled['candidate'], kw)
                    torch.cuda.synchronize()
                    audit = candidate.AUDIT
                    candidate.AUDIT = None
                    check(label + '_compiled_exact', equal_tree(expected, actual))
                    check(label + '_reference_AA', equal_tree(expected, run(compiled['reference'], kw)))
                    check(label + '_candidate_repeat', equal_tree(actual, run(compiled['candidate'], kw)))
                    rq, rk, rv = audit['reference']['inputs']
                    refout = audit['reference']['output']
                    segments = audit['segments']
                    count = 10 if phase == 'prefill' else 1
                    check(label + '_segment_count', len(segments) == count, count=len(segments))
                    offset = 0
                    for i, record in enumerate(segments):
                        q, k, v = record['inputs']
                        end = offset + q.shape[1]
                        check(label + '_operands_' + str(i), bits(q, rq[:, offset:end])
                              and bits(k, rk[:, :k.shape[1]]) and bits(v, rv[:, :v.shape[1]]))
                        check(label + '_readonly_' + str(i), equal_tree(record['inputs'], record['after'])
                              and record['fresh'])
                        check(label + '_attention_exact_' + str(i), bits(record['output'], refout[:, offset:end]))
                        with torch._subclasses.fake_tensor.FakeTensorMode() as fm:
                            fake = candidate.segment(*[fm.from_tensor(x) for x in (q, k, v)],
                                                     attention.softmax_scale, record['causal'])
                            real = record['output']
                            metadata_ok = fake.shape == real.shape and fake.dtype == real.dtype and fake.device == real.device
                            metadata_ok = metadata_ok and tuple(fake.stride()) == record['stride']
                        check(label + '_fake_' + str(i), metadata_ok)
                        offset = end
                    check(label + '_all_queries_checked', offset == n)
                    case['segment_metadata'] = [{k: row[k] for k in ('metadata', 'causal', 'stride')} for row in segments]
                    del audit, segments, rq, rk, rv, refout, q, k, v, real, record

                    hidden = kw['hidden_states']
                    original_hidden = hidden.clone()
                    hidden.mul_(0.875)
                    changed = run(compiled['reference'], kw)
                    check(label + '_changed_compiled_exact', equal_tree(changed, run(compiled['candidate'], kw)))
                    check(label + '_changed_not_stale', not bits(expected[0], changed[0]))
                    hidden.copy_(original_hidden)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graphs = []
                    with torch.cuda.stream(stream):
                        for arm in ('reference', 'candidate'):
                            for _ in range(3):
                                run(compiled[arm], kw)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream):
                                graph_output = run(compiled[arm], kw)
                            graphs.append((graph, graph_output))
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    hidden.mul_(0.875)
                    for arm, (graph, graph_output) in zip(('reference', 'candidate'), graphs, strict=True):
                        graph.replay()
                        torch.cuda.synchronize()
                        check(label + '_' + arm + '_changed_replay', equal_tree(changed, graph_output))
                    hidden.copy_(original_hidden)
                    for arm, (graph, graph_output) in zip(('reference', 'candidate'), graphs, strict=True):
                        graph.replay()
                        torch.cuda.synchronize()
                        check(label + '_' + arm + '_restored_replay', equal_tree(expected, graph_output))
                    check(label + '_inputs_unchanged', all(equal_tree(kw[k], v) for k, v in saved_inputs.items()))
                    if phase == 'prefill':
                        cache[0].update(clone_tree(expected[1]))
                    del graphs, graph, graph_output, expected, actual, changed, saved_inputs, original_hidden
                    save()
            torch.cuda.synchronize()
            from torch._dynamo.utils import counters
            result['graph_breaks'] = dict(counters['graph_break'])
            check('no_graph_breaks', not counters['graph_break'])
            files = []
            for path in Path(os.environ['TORCHINDUCTOR_CACHE_DIR']).rglob('*.py'):
                source = path.read_text()
                if 'torch.ops.qwen21_functional_probe.segment.default' in source:
                    files.append(dict(path=str(path), sha256=hashlib.sha256(source.encode()).hexdigest(),
                        segment_calls=source.count('torch.ops.qwen21_functional_probe.segment.default('),
                        mutable_aiter_calls=source.count('torch.ops.aiter.mha_fwd.default(')
                                            + source.count('torch.ops.aiter.fmha_v3_fwd.default(')))
            result['generated_modules'] = files
            check('functional_generated_calls', bool(files) and all(x['segment_calls'] > 0 for x in files))
            check('no_mutable_aiter_calls', all(x['mutable_aiter_calls'] == 0 for x in files))
            check('gpu_health_after', torch.ones(128, device='cuda').sum().item() == 128)
            result['status'] = 'completed'
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

