# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Profile one functional-AITER configuration using the production diffuse loop."""

from contextlib import ExitStack, nullcontext
import functools
import gc
import hashlib
import statistics
import time
from unittest.mock import patch

import torch


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def run(model, ctx, checkpoint, output_dir, check, result, save):
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from torch._dynamo.utils import counters
    from vllm_omni.diffusion.models.qwen_image_21 import decode_graph as dg
    from vllm_omni.diffusion.models.qwen_image_21.cfg_parallel import QwenImage21CFGParallelMixin
    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline

    class Denoiser(QwenImage21CFGParallelMixin):
        interrupt = False

    holder = Denoiser()
    holder.transformer = model
    holder.device = torch.device('cuda')
    holder.set_progress_bar_config(disable=True)
    holder.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(checkpoint), subfolder='scheduler', local_files_only=True)
    scheduler_config = holder.scheduler.config
    args = {k: ctx[k] for k in (
        'prompt_embeds', 'prompt_embeds_mask', 'negative_prompt_embeds', 'negative_prompt_embeds_mask',
        'img_mask', 'negative_img_mask', 'image_latents', 'img_shapes', 'do_true_cfg')}
    manager = model._decode_graph_manager
    report = result['profile'] = dict(runs=[], backend='FLASH_ATTN/AITER', source='main with validated functional AITER boundary',
        steps=3, schedule='first three of the saved 40-step schedule', blocks=32,
        target_tokens=ctx['latents'].shape[1], shapes=ctx['img_shapes'],
        scope='DiT only; encoded context to step-3 latent, excluding encoder/VAE/HTTP')
    original_predict = holder.predict_noise
    original_capture = dg.QwenImage21DecodeGraphEntry.capture

    def one(label, profiled):
        gc.collect()
        manager.release_idle_entries()
        check(label + '_empty_graphs', not manager.entries and not manager._owners)
        holder.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        schedule, _ = QwenImage21Pipeline.prepare_timesteps(holder, 40, None, ctx['latents'].shape[1])
        check(label + '_saved_schedule', exact(schedule, ctx['timesteps']))
        initial = ctx['latents'].clone()
        predictions, captures = [], []

        def predict(**kwargs):
            step = len(predictions) // 2
            phase = ('prefill', 'capture_decode', 'steady_decode')[step]
            branch = kwargs['cache_branch']
            start = event()
            wall = time.perf_counter()
            with torch.profiler.record_function('qwen21/' + phase + '/' + branch):
                output = original_predict(**kwargs)
            finish = event()
            predictions.append(dict(step=step, phase=phase, branch=branch, start=start, end=finish,
                                    host_submit_ms=(time.perf_counter() - wall) * 1000))
            return output

        def capture(entry, body):
            calls = 0

            def mark_body():
                nonlocal calls
                calls += 1
                phase = 'warmup' + str(calls) if calls <= 2 else 'capture_body'
                with torch.profiler.record_function('qwen21/' + phase + '/' + entry.branch):
                    return body()

            start = event()
            wall = time.perf_counter()
            with torch.profiler.record_function('qwen21/graph_setup/' + entry.branch):
                value = original_capture(entry, mark_body)
            captures.append(dict(branch=entry.branch, body_calls=calls, start=start, end=event(),
                                 wall_ms=(time.perf_counter() - wall) * 1000))
            return value

        with ExitStack() as stack:
            stack.enter_context(patch.object(holder, 'predict_noise', predict))
            stack.enter_context(patch.object(dg.QwenImage21DecodeGraphEntry, 'capture', capture))
            if profiled:
                for index, block in enumerate(model.transformer_blocks):
                    original = block.forward

                    def wrapper(original, index):
                        @functools.wraps(original)
                        def marked(*pos, **kw):
                            with torch.profiler.record_function(f'qwen21/block/{index:02d}'):
                                return original(*pos, **kw)
                        return marked

                    stack.enter_context(patch.object(block, 'forward', wrapper(original, index)))
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, with_stack=False, profile_memory=False,
            ) if profiled else nullcontext()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = counters['stats']['unique_graphs']
            with profiler as p:
                start = event()
                wall = time.perf_counter()
                with torch.profiler.record_function('qwen21/three_steps'):
                    final = holder.diffuse(**args, latents=initial, timesteps=schedule[:3], true_cfg_scale=4.0,
                        attention_kwargs={'attn_path': 'auto'}, additional_transformer_kwargs={'return_dict': False})
                end = event()
                end.synchronize()
                wall_ms = (time.perf_counter() - wall) * 1000
            if profiled:
                path = output_dir / 'stages-trace.json'
                p.export_chrome_trace(str(path))
                report['trace'] = str(path)
                report['trace_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()

        check(label + '_six_calls', [x['branch'] for x in predictions] == ['cond', 'uncond'] * 3)
        check(label + '_two_captures', len(captures) == 2 and all(x['body_calls'] == 3 for x in captures)
              and len(manager.entries) == 2 and all(e.captures == 1 and e.graph is not None for e in manager.entries.values()))
        check(label + '_finite', bool(torch.isfinite(final).all()))
        for x in predictions + captures:
            x['gpu_elapsed_ms'] = x.pop('start').elapsed_time(x.pop('end'))
        cpu = final.detach().cpu()
        torch.save(cpu, output_dir / (label + '-final.pt'))
        row = dict(label=label, profiled=profiled, wall_ms=wall_ms, gpu_elapsed_ms=start.elapsed_time(end),
                   predictions=predictions, captures=captures,
                   new_graphs=counters['stats']['unique_graphs'] - before,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   final_sha256=hashlib.sha256(cpu.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        row['stages_ms'] = {phase: sum(x['gpu_elapsed_ms'] for x in predictions if x['phase'] == phase)
                            for phase in ('prefill', 'capture_decode', 'steady_decode')}
        row['graph_setup_wall_ms'] = sum(x['wall_ms'] for x in captures)
        report['runs'].append(row)
        print('RUN ' + str({k: row[k] for k in ('label', 'wall_ms', 'new_graphs', 'stages_ms', 'graph_setup_wall_ms')}), flush=True)
        save()
        manager.release_idle_entries()
        check(label + '_released', not manager.entries and not manager._owners)
        return cpu

    reference = one('compile_warmup', False)
    for label, profiled in (('timed_A', False), ('timed_B', False), ('profiled', True)):
        observed = one(label, profiled)
        check(label + '_matches_warmup', exact(reference, observed))
        check(label + '_no_recompile', report['runs'][-1]['new_graphs'] == 0)
    timed = [x for x in report['runs'] if x['label'].startswith('timed_')]
    report['timing_medians'] = dict(
        wall_ms=statistics.median(x['wall_ms'] for x in timed),
        stages_ms={phase: statistics.median(x['stages_ms'][phase] for x in timed)
                   for phase in ('prefill', 'capture_decode', 'steady_decode')},
        graph_setup_wall_ms=statistics.median(x['graph_setup_wall_ms'] for x in timed),
    )
    from analyze import analyze
    report['analysis'] = analyze(output_dir / 'stages-trace.json', output_dir / 'trace-analysis.json')
    check('no_sdpa_in_trace', report['analysis']['sdpa_cpu_events'] == 0)
    save()
