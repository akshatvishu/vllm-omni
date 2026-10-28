"""Production diffuse loop, exact trajectory gate, then bounded paired timing."""
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def run(model, ctx, checkpoint, output_dir, check, result, save):
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from torch._dynamo.utils import counters
    from vllm_omni.diffusion.models.qwen_image_21 import decode_graph as dg
    from vllm_omni.diffusion.models.qwen_image_21.cfg_parallel import QwenImage21CFGParallelMixin
    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline

    report = result['denoise'] = {'runs': [], 'artifacts': {}, 'summary': None}
    entry_type = dg.QwenImage21DecodeGraphEntry
    manager = model._decode_graph_manager

    def native_entry(**kwargs):
        kwargs['retain_attention_buffers'] = False
        return entry_type(**kwargs)

    # Use the production mixin and scheduler methods unchanged. The lightweight
    # holder omits loading encoders/VAE, which are outside this measurement.
    class Denoiser(QwenImage21CFGParallelMixin):
        interrupt = False

    holder = Denoiser()
    holder.transformer = model
    holder.device = torch.device('cuda')
    holder.set_progress_bar_config(disable=True)
    holder.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(checkpoint), subfolder='scheduler', local_files_only=True)
    scheduler_config = holder.scheduler.config
    schedule, _ = QwenImage21Pipeline.prepare_timesteps(holder, 40, None, ctx['latents'].shape[1])
    check('saved_40_step_schedule', len(schedule) == 40 and exact(schedule, ctx['timesteps']))
    check('actual_production_loop', holder.diffuse.__func__ is QwenImage21CFGParallelMixin.diffuse)
    args = {k: ctx[k] for k in (
        'prompt_embeds', 'prompt_embeds_mask', 'negative_prompt_embeds', 'negative_prompt_embeds_mask',
        'img_mask', 'negative_img_mask', 'image_latents', 'img_shapes', 'do_true_cfg')}
    report['configuration'] = dict(steps=40, cfg_scale=4.0, cfg_normalize=False, target_tokens=ctx['latents'].shape[1],
        img_shapes=ctx['img_shapes'], backend='TORCH_SDPA', dtype='bfloat16', tp=1, sp=1,
        timing_pairs=6, baseline_control_pairs=2, scope='Production diffuse() from saved encoded context to final latent; fresh prefill/cache/graph capture each run; prior-request cleanup outside timing')
    tensor_dir = output_dir / 'tensors'
    tensor_dir.mkdir()

    def artifact(label, tensor):
        path = tensor_dir / (label + '.pt')
        cpu = tensor.detach().cpu()
        torch.save(cpu, path)
        report['artifacts'][label] = dict(path=str(path), content_sha256=digest(cpu),
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=list(cpu.shape), dtype=str(cpu.dtype))

    def one(label, arm, diagnostic=False, timed=False):
        # Clear only this experiment's dead entries. Setup within diffuse,
        # including new entry allocation and capture, remains in the interval.
        manager.release_idle_entries()
        gc.collect()
        check(label + '_starts_empty', not manager.entries and not manager._owners)
        holder.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        timesteps, _ = QwenImage21Pipeline.prepare_timesteps(holder, 40, None, ctx['latents'].shape[1])
        initial = ctx['latents'].clone()
        check(label + '_same_schedule', exact(timesteps, ctx['timesteps']))
        predictions, trajectory, references = [], [], []
        observed_calls = []
        original_predict = holder.predict_noise
        original_step = holder.scheduler_step_maybe_with_cfg

        def observe_predict(**kwargs):
            value = original_predict(**kwargs)
            predictions.append(value[:, -initial.shape[1]:].clone())
            observed_calls.append(kwargs['cache_branch'])
            if not references:
                references.append(kwargs['kv_cache'])
            return value

        def observe_step(*pos, **kw):
            value = original_step(*pos, **kw)
            trajectory.append(value.clone())
            return value

        with patch.object(dg, 'QwenImage21DecodeGraphEntry', native_entry if arm == 'B0' else entry_type):
            with (patch.object(holder, 'predict_noise', observe_predict) if diagnostic else nullcontext()):
                with (patch.object(holder, 'scheduler_step_maybe_with_cfg', observe_step) if diagnostic else nullcontext()):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    before_graphs = counters['stats']['unique_graphs']
                    host_start = time.perf_counter_ns()
                    start.record()
                    final = holder.diffuse(**args, latents=initial, timesteps=timesteps, true_cfg_scale=4.0,
                        attention_kwargs={'attn_path': 'auto'}, additional_transformer_kwargs={'return_dict': False})
                    end.record()
                    end.synchronize()
                    wall_ms = (time.perf_counter_ns() - host_start) / 1e6
                    elapsed_ms = start.elapsed_time(end)
        entries = list(manager.entries.values())
        check(label + '_both_graphs_captured_once', len(entries) == 2 and
              all(e is not None and e.graph is not None and e.captures == 1 for e in entries))
        check(label + '_buffer_policy', all((e.packed_k is not None) == (arm == 'P1') for e in entries))
        check(label + '_finite_final', bool(torch.isfinite(final).all()))
        if timed:
            check(label + '_no_new_compilation', counters['stats']['unique_graphs'] == before_graphs)
        row = dict(label=label, arm=arm, diagnostic=diagnostic, timed=timed,
            wall_ms=wall_ms, elapsed_device_ms=elapsed_ms,
            final_sha256=digest(final), final_shape=list(final.shape), final_stride=list(final.stride()),
            unique_graphs_before=before_graphs, unique_graphs_after=counters['stats']['unique_graphs'],
            captures={e.branch: e.captures for e in entries})
        if diagnostic:
            check(label + '_80_branch_predictions', observed_calls == ['cond', 'uncond'] * 40)
            check(label + '_40_scheduler_updates', len(trajectory) == 40 and holder.scheduler.step_index == 40)
            # Both saved tensors are small compared with the model; no snapshots
            # are retained or callbacks installed during timing.
            predictions = torch.stack(predictions).cpu()
            trajectory = torch.stack(trajectory).cpu()
            artifact(label + '_predictions', predictions)
            artifact(label + '_latents', trajectory)
            row['prefix_sha256'] = {f'{i}/{branch}/{kind}': digest(cache[branch][kind])
                for i, cache in enumerate(references[0]) for branch in ('cond', 'uncond') for kind in ('key', 'value')}
            references.clear()
        artifact(label + '_final', final)
        report['runs'].append(row)
        print('RUN ' + json.dumps({k: v for k, v in row.items() if k != 'prefix_sha256'}), flush=True)
        # Diagnostic wrapper references are now released so ordinary owner-based
        # cleanup can free the entries. No retained buffer crosses requests.
        del entries, initial
        gc.collect()
        manager.release_idle_entries()
        check(label + '_request_released', not manager.entries and not manager._owners)
        save()
        return final.cpu(), predictions, trajectory, row

    # Compile/warm both paths and test independent complete trajectories before
    # collecting any performance samples. All prefill values are unmodified.
    ref, ref_pred, ref_lat, ref_row = one('diagnostic_B0', 'B0', diagnostic=True)
    for label, arm in [('diagnostic_P1', 'P1'), ('diagnostic_B0_repeat', 'B0')]:
        final, pred, lat, row = one(label, arm, diagnostic=True)
        pred_equal = [exact(x, y) for x, y in zip(ref_pred, pred)]
        lat_equal = [exact(x, y) for x, y in zip(ref_lat, lat)]
        row['prediction_equal_by_step_branch'] = pred_equal
        row['latent_equal_by_step'] = lat_equal
        check(label + '_every_prediction_exact', len(pred_equal) == 80 and all(pred_equal),
              passed_predictions=sum(pred_equal), total_predictions=80)
        check(label + '_every_latent_exact', len(lat_equal) == 40 and all(lat_equal),
              passed_latents=sum(lat_equal), total_latents=40)
        check(label + '_final_exact', exact(ref, final) and ref.stride() == final.stride())
        check(label + '_prefixes_exact', row['prefix_sha256'] == ref_row['prefix_sha256'])
    del ref_pred, ref_lat, pred, lat

    def timed(label, arm):
        final, _, _, row = one(label, arm, timed=True)
        check(label + '_final_matches_baseline', exact(ref, final) and ref.stride() == final.stride())
        return row

    aa, ab = [], []
    # A/A controls bracket the alternating pairs so ordinary drift is visible.
    aa.append([timed('AA0_a', 'B0'), timed('AA0_b', 'B0')])
    for index in range(6):
        arms = ['B0', 'P1'] if index % 2 == 0 else ['P1', 'B0']
        pair = {'order': arms}
        for arm in arms:
            pair[arm] = timed(f'pair{index}_{arm}', arm)
        ab.append(pair)
    aa.append([timed('AA1_a', 'B0'), timed('AA1_b', 'B0')])
    summaries = {}
    for metric in ('wall_ms', 'elapsed_device_ms'):
        controls = [abs(a[metric] - b[metric]) for a, b in aa]
        savings = [p['B0'][metric] - p['P1'][metric] for p in ab]
        percentages = [100 * (p['B0'][metric] - p['P1'][metric]) / p['B0'][metric] for p in ab]
        summaries[metric] = dict(native_median=statistics.median(p['B0'][metric] for p in ab),
            p1_median=statistics.median(p['P1'][metric] for p in ab),
            paired_saving_ms=statistics.median(savings), paired_improvement_pct=statistics.median(percentages),
            positive_pairs=sum(x > 0 for x in savings), pairs=6, baseline_control_max_abs_ms=max(controls),
            screen_passed=sum(x > 0 for x in savings) >= 5 and statistics.median(savings) > max(controls)
            and statistics.median(percentages) >= 1.0)
    report.update(summary=summaries, AA=aa, AB=ab,
        dynamo_counters={k: dict(counters[k]) for k in ('frames', 'stats', 'graph_break')},
        decision='keep' if summaries['wall_ms']['screen_passed'] else 'no demonstrated denoising benefit under the predeclared screen')
    save()
