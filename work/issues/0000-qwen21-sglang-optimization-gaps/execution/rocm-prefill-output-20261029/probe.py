# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded real-operand AITER prefill output-buffer experiment."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import time
import traceback
from unittest.mock import patch

import torch


def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def measure(fn, operands, repeats=10):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(repeats):
        output = fn(*operands)
    end.record()
    torch.cuda.synchronize()
    return dict(gpu_ms=start.elapsed_time(end) / repeats,
                wall_ms=(time.perf_counter() - wall) * 1000 / repeats), output


def profile(fn, operands, path):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as p:
        fn(*operands)
        torch.cuda.synchronize()
    p.export_chrome_trace(str(path))
    events = p.events()
    return dict(
        kernels=[dict(name=e.name, duration_us=e.device_time_total)
                 for e in events if e.device_type == torch.autograd.DeviceType.CUDA],
        cat_calls=sum(e.count for e in p.key_averages() if e.key == 'aten::cat'),
        sdpa_calls=sum(e.count for e in p.key_averages() if 'scaled_dot_product' in e.key),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--context', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', checks=[], cases=[], scope='first-layer AITER prefill attention')

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
        from candidate import direct_output

        torch.cuda.set_device(0)
        prop = torch.cuda.get_device_properties(0)
        check('gfx942', torch.version.hip is not None and 'gfx942' in prop.gcnArchName,
              device=prop.name, arch=prop.gcnArchName)
        check('aiter_binding', fa.IS_AITER and fa.flash_attn_func is mha.flash_attn_func)
        result['stack'] = dict(torch=torch.__version__, torch_git=torch.version.git_version,
                               hip=torch.version.hip, aiter_file=inspect.getfile(mha),
                               aiter_sha256=hashlib.sha256(Path(inspect.getfile(mha)).read_bytes()).hexdigest())
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
              name=attention.attn_backend.get_name())

        class Captured(Exception):
            pass

        with set_forward_context(vllm_config=vcfg, omni_diffusion_config=od), torch.inference_mode(), patch(
            'torch.nn.functional.scaled_dot_product_attention',
            side_effect=AssertionError('SDPA is forbidden in this experiment'),
        ):
            model.do_true_cfg = True
            for branch in ('cond', 'uncond'):
                operands = []

                def capture(module, args):
                    operands.extend(args)
                    raise Captured()

                handle = attention.register_forward_pre_hook(capture)
                negative = branch == 'uncond'
                try:
                    model(hidden_states=torch.cat([ctx['image_latents'], ctx['latents']], dim=1),
                        encoder_hidden_states=ctx['negative_prompt_embeds' if negative else 'prompt_embeds'],
                        timestep=ctx['timesteps'][0].expand(1).to(ctx['latents'].dtype) / 1000,
                        img_shapes=ctx['img_shapes'],
                        img_mask=ctx['negative_img_mask' if negative else 'img_mask'],
                        encoder_hidden_states_mask=ctx['negative_prompt_embeds_mask' if negative else 'prompt_embeds_mask'],
                        kv_cache=QwenImage21KVCache([{}]), cache_branch=branch, return_dict=False)
                except Captured:
                    pass
                finally:
                    handle.remove()
                check(branch + '_captured', len(operands) == 4)
                q, k, v, metadata = operands
                check(branch + '_piecewise', metadata.full_attn_spans is not None and metadata.attn_mask is None)
                spans = metadata.full_attn_spans[0]
                flat = [x for span in spans for x in span]
                case = dict(branch=branch, shape=list(q.shape), spans=spans,
                            strides=[list(x.stride()) for x in (q, k, v)],
                            offsets=[x.storage_offset() for x in (q, k, v)],
                            segments=[list(x) for x in build_segments(spans, 0, q.shape[1])])
                result['cases'].append(case)
                original = [x.clone() for x in (q, k, v)]

                def native(q, k, v):
                    return attention(q, k, v, metadata)

                def candidate(q, k, v):
                    return direct_output(q, k, v, flat, attention.softmax_scale)

                reference = native(q, k, v)
                check(branch + '_native_AA', bits(reference, native(q, k, v)))
                actual = candidate(q, k, v)
                check(branch + '_eager_exact', bits(reference, actual))
                check(branch + '_output_metadata', actual.dtype == reference.dtype and actual.shape == reference.shape
                      and actual.stride() == reference.stride() and actual.is_contiguous(), stride=list(actual.stride()))
                with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
                    fake = direct_output(*[mode.from_tensor(x) for x in (q, k, v)], flat, attention.softmax_scale)
                    fake_ok = fake.shape == actual.shape and fake.dtype == actual.dtype and fake.stride() == actual.stride()
                check(branch + '_fake_metadata', fake_ok)
                for mode_name, funcs in (
                    ('eager', (native, candidate)),
                    ('compiled', tuple(torch.compile(fn, dynamic=od.diffusion_compile_dynamic,
                        options={'triton.cudagraphs': False, 'triton.cudagraph_trees': False})
                        for fn in (native, candidate))),
                ):
                    baseline, optimized = funcs
                    t0 = time.perf_counter()
                    for _ in range(3):
                        baseline(q, k, v)
                        optimized(q, k, v)
                    torch.cuda.synchronize()
                    setup = time.perf_counter() - t0
                    check(branch + '_' + mode_name + '_exact', bits(baseline(q, k, v), optimized(q, k, v)))
                    check(branch + '_' + mode_name + '_baseline_matches_eager', bits(reference, baseline(q, k, v)))
                    q.mul_(0.875)
                    check(branch + '_' + mode_name + '_changed_input', bits(baseline(q, k, v), optimized(q, k, v)))
                    q.copy_(original[0])
                    controls = []
                    for _ in range(2):
                        a_time, _ = measure(baseline, (q, k, v))
                        b_time, _ = measure(baseline, (q, k, v))
                        controls.append(dict(a=a_time, b=b_time))
                    pairs = []
                    for i in range(8):
                        order = (('baseline', baseline), ('candidate', optimized))
                        if i % 2:
                            order = tuple(reversed(order))
                        row = {}
                        for name, fn in order:
                            row[name], last = measure(fn, (q, k, v))
                        pairs.append(row)
                    check(branch + '_' + mode_name + '_post_timing_exact', bits(baseline(q, k, v), optimized(q, k, v)))
                    summary = {}
                    for metric in ('gpu_ms', 'wall_ms'):
                        reductions = [100 * (1 - row['candidate'][metric] / row['baseline'][metric]) for row in pairs]
                        savings = [row['baseline'][metric] - row['candidate'][metric] for row in pairs]
                        aa = max(abs(row['a'][metric] - row['b'][metric]) for row in controls)
                        summary[metric] = dict(baseline=statistics.median(row['baseline'][metric] for row in pairs),
                            candidate=statistics.median(row['candidate'][metric] for row in pairs),
                            reduction_pct=statistics.median(reductions), saving=statistics.median(savings),
                            min_pct=min(reductions), max_pct=max(reductions), aa_max_delta=aa,
                            signal_exceeds_AA=statistics.median(savings) > aa)
                    case[mode_name] = dict(setup_s=setup, pairs=pairs, controls=controls, summary=summary)
                    print(json.dumps(dict(branch=branch, mode=mode_name, summary=summary)), flush=True)
                    save()
                    if mode_name == 'compiled':
                        traces = {}
                        for name, fn in (('baseline', baseline), ('candidate', optimized)):
                            traces[name] = profile(fn, (q, k, v), a.output / (branch + '-' + name + '-trace.json'))
                        case['traces'] = traces
                        check(branch + '_no_sdpa_trace', all(x['sdpa_calls'] == 0 for x in traces.values()))
                        check(branch + '_candidate_no_cat', traces['candidate']['cat_calls'] == 0)
                        # Graph replay is a changed-input correctness check, not the timing boundary.
                        graphs = []
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for fn in funcs:
                                fn(q, k, v)
                                fn(q, k, v)
                        torch.cuda.current_stream().wait_stream(stream)
                        for fn in funcs:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                out = fn(q, k, v)
                            graphs.append((graph, out))
                        q.mul_(0.875)
                        for graph, _ in graphs:
                            graph.replay()
                        torch.cuda.synchronize()
                        check(branch + '_changed_graph_replay', bits(graphs[0][1], graphs[1][1])
                              and bits(graphs[0][1], native(q, k, v)))
                        q.copy_(original[0])
                        del graphs
                check(branch + '_inputs_preserved', all(bits(x, y) for x, y in zip((q, k, v), original, strict=True)))
                save()
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
