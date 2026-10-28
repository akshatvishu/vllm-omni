# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Outer request timing; no wrappers inside compiled transformer blocks."""
import functools
import json
import os
from pathlib import Path
import time

import torch

ACTIVE = None
SEQUENCE = 0
ROOT = Path(os.environ['QWEN_EXPERIMENT_RUN_DIR'])
OUTPUT = ROOT / 'production-timing'


def event():
    value = torch.cuda.Event(enable_timing=True)
    value.record()
    return value


def install_pipeline(module, identify):
    cls = module.QwenImage21Pipeline
    prepare_original = cls._prepare_generation_context
    step_original = cls._denoise_step_homogeneous
    scheduler_original = cls.step_scheduler
    decode_original = cls._decode_latents

    @functools.wraps(prepare_original)
    def prepare(self, **kwargs):
        global ACTIVE, SEQUENCE
        if not (ROOT / 'enable-production-timing').exists():
            return prepare_original(self, **kwargs)
        assert ACTIVE is None, 'Only serial requests supported by this probe'
        SEQUENCE += 1
        ACTIVE = dict(sequence=SEQUENCE, case=identify(kwargs), steps=[], captures=[], events={},
                      profiled=(ROOT / 'enable-production-profile').exists(),
                      request_phase=(ROOT / 'production-phase.txt').read_text(),
                      prepare_start_time_ns=time.time_ns(), phase='prepare')
        ACTIVE['prepare_wall_start'] = time.perf_counter()
        ACTIVE['events']['prepare_start'] = event()
        result = prepare_original(self, **kwargs)
        ACTIVE['events']['prepare_end'] = event()
        ACTIVE['prepare_wall_enqueue_ms'] = (time.perf_counter() - ACTIVE['prepare_wall_start']) * 1000
        ACTIVE['phase'] = 'ready'
        return result

    @functools.wraps(step_original)
    def step(self, batch):
        if ACTIVE is None:
            return step_original(self, batch)
        assert len(batch.states) == 1
        state = batch.states[0]
        index = state.step_index
        ACTIVE['request_id'] = state.request_id
        ACTIVE['steps'].append(index)
        if index == 0:
            ACTIVE['denoise_wall_start'] = time.perf_counter()
            ACTIVE['events']['denoise_start'] = event()
        elif index == 1:
            ACTIVE['events']['decode_start'] = event()
        elif index == 2:
            ACTIVE['events']['steady_start'] = event()
        ACTIVE['phase'] = f'step-{index}'
        if ACTIVE['profiled'] and index == 2:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                result = step_original(self, batch)
            ACTIVE['profile'] = profile
            return result
        return step_original(self, batch)

    @functools.wraps(scheduler_original)
    def scheduler(self, state, noise_pred, **kwargs):
        result = scheduler_original(self, state, noise_pred, **kwargs)
        if ACTIVE is not None:
            if state.step_index == 1:
                ACTIVE['events']['prefill_end'] = event()
            if state.step_index == len(state.timesteps):
                end = event()
                ACTIVE['events']['denoise_end'] = end
                # One synchronization at the stage boundary, never per step.
                end.synchronize()
                ACTIVE['denoise_wall_ms'] = (time.perf_counter() - ACTIVE['denoise_wall_start']) * 1000
                ACTIVE['denoise_end_time_ns'] = time.time_ns()
        return result

    @functools.wraps(decode_original)
    def decode(self, *args, **kwargs):
        global ACTIVE
        if ACTIVE is None:
            return decode_original(self, *args, **kwargs)
        active = ACTIVE
        active['phase'] = 'vae'
        start = time.perf_counter()
        active['events']['vae_start'] = event()
        result = decode_original(self, *args, **kwargs)
        active['events']['vae_end'] = event()
        active['events']['vae_end'].synchronize()
        active['vae_wall_ms'] = (time.perf_counter() - start) * 1000
        assert active['steps'] == list(range(40)), active['steps']
        ev = active.pop('events')
        spans = {'prepare':('prepare_start','prepare_end'), 'denoise40':('denoise_start','denoise_end'),
                 'prefill':('denoise_start','prefill_end'), 'decode39':('decode_start','denoise_end'),
                 'steady38':('steady_start','denoise_end'), 'vae':('vae_start','vae_end')}
        active['gpu_span_ms'] = {name:ev[a].elapsed_time(ev[b]) for name,(a,b) in spans.items()}
        for capture in active['captures']:
            capture['gpu_span_ms'] = capture.pop('start_event').elapsed_time(capture.pop('end_event'))
        active['graph_setup_wall_ms'] = sum(x['wall_ms'] for x in active['captures'])
        active['graph_setup_gpu_span_ms'] = sum(x['gpu_span_ms'] for x in active['captures'])
        active['decode39_minus_setup_gpu_span_ms'] = active['gpu_span_ms']['decode39'] - active['graph_setup_gpu_span_ms']
        active['timing_eligible'] = active['request_phase'] == 'measured' and not active['profiled']
        active['scope'] = 'Event spans include host submission gaps. Denoise wall waits once after final scheduler step. No per-step synchronization. Profiled rows excluded from timing comparisons.'
        profile = active.pop('profile', None)
        active.pop('prepare_wall_start')
        active.pop('denoise_wall_start')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        name = f"{active['sequence']:03d}-{active['case']}"
        if profile is not None:
            profile.export_chrome_trace(str(OUTPUT / (name + '-decode-step-trace.json')))
        (OUTPUT / (name + '.json')).write_text(json.dumps(active, indent=2) + '\n')
        ACTIVE = None
        return result

    cls._prepare_generation_context = prepare
    cls._denoise_step_homogeneous = step
    cls.step_scheduler = scheduler
    cls._decode_latents = decode


def install_graph(module):
    cls = module.QwenImage21DecodeGraphEntry
    original = cls.capture

    @functools.wraps(original)
    def capture(self, body):
        if ACTIVE is None:
            return original(self, body)
        record = dict(branch=self.branch, phase=ACTIVE['phase'], start_event=event())
        start = time.perf_counter()
        result = original(self, body)
        record['wall_ms'] = (time.perf_counter() - start) * 1000
        record['end_event'] = event()
        ACTIVE['captures'].append(record)
        return result

    cls.capture = capture
