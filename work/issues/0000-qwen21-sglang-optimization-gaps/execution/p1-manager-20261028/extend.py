"""Attention observation and bounded timing after the copy screen passes.

Observe the existing generated Python wrappers, not a recompiled model with
hooks. Only the wrapper's private torch namespace is proxied. Its actual SDPA
call executes unchanged. Restore namespaces before uninstrumented HIP replay.
"""
from contextlib import contextmanager
from pathlib import Path
import statistics
import time

import torch


class Proxy:
    def __init__(self, real, **overrides):
        self.real = real
        self.overrides = overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.real, name)


@contextmanager
def observe_attention(callback):
    from torch._inductor.codecache import PyCodeCache

    original = torch.ops.aten._scaled_dot_product_flash_attention.default

    def observed(q, k, v, *args, **kwargs):
        output = original(q, k, v, *args, **kwargs)
        callback(q, k, v, output[0])
        return output

    patched = []
    proxy = Proxy(torch, ops=Proxy(torch.ops, aten=Proxy(torch.ops.aten,
        _scaled_dot_product_flash_attention=Proxy(torch.ops.aten._scaled_dot_product_flash_attention, default=observed))))
    for module in PyCodeCache.modules:
        file = getattr(module, '__file__', None)
        if file and '_scaled_dot_product_flash_attention.default(' in Path(file).read_text():
            patched.append((module, module.torch))
            module.torch = proxy
    if not patched:
        raise RuntimeError('No generated attention wrappers to observe')
    try:
        yield
    finally:
        for module, original_torch in patched:
            module.torch = original_torch


def run(model, native, candidate, inputs, check, bits, result, save):
    from torch._dynamo.utils import counters

    graph_count = counters['stats']['unique_graphs']
    observations = []
    for step, branch, kw in inputs:
        nout = native.try_decode(**kw)
        pout = candidate.try_decode(**kw)
        ne = next(e for e in native.entries.values() if e.branch == branch)
        pe = next(e for e in candidate.entries.values() if e.branch == branch)
        snapshots = []

        def record(*tensors):
            snapshots.append([(t.clone(), tuple(t.stride())) for t in tensors])

        rows = []

        def compare(*tensors):
            i = len(rows)
            if i >= len(snapshots):
                raise RuntimeError('Unexpected extra candidate attention call')
            rows.append({'layer': i, 'exact': [bits(t, old) for t, (old, stride) in zip(tensors, snapshots[i])],
                         'same_strides': [tuple(t.stride()) == stride for t, (old, stride) in zip(tensors, snapshots[i])]})

        with torch.inference_mode(False), torch.no_grad():
            with observe_attention(record):
                observed_native = model._decode_graph_forward(ne)
            with observe_attention(compare):
                observed_candidate = model._decode_graph_forward(pe)
        check(f'observed_s{step}_{branch}_all_32_attention_QKVO', len(rows) == len(snapshots) == 32 and
              all(all(r['exact']) and all(r['same_strides']) for r in rows), calls=len(rows))
        check(f'observed_s{step}_{branch}_matches_graph_outputs', bits(observed_native, nout) and bits(observed_candidate, pout))
        observations.append({'step': step, 'branch': branch, 'layers': rows})
        del snapshots, ne, pe, nout, pout, observed_native, observed_candidate
    check('observation_did_not_recompile', counters['stats']['unique_graphs'] == graph_count)
    result['attention_observation'] = observations
    result['attention_scope'] = 'Actual generated regional wrappers without HIP capture; Q/K/V/output exact at all layers and both changed inputs; whole observed outputs match uninstrumented graph outputs.'
    save()

    latest = {branch: kw for step, branch, kw in inputs if step == 2}

    def invoke(manager):
        for branch in ('cond', 'uncond'):
            manager.try_decode(**latest[branch])

    for _ in range(5):
        invoke(native)
        invoke(candidate)
    torch.cuda.synchronize()

    def timed(manager):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        host_start = time.perf_counter_ns()
        start.record()
        invoke(manager)
        end.record()
        end.synchronize()
        return {'device_ms': start.elapsed_time(end), 'wall_ms': (time.perf_counter_ns() - host_start) / 1e6}

    aa = [(timed(native), timed(native)) for _ in range(5)]
    pairs = []
    deadline = time.monotonic() + 60
    for i in range(10):
        if time.monotonic() > deadline:
            raise RuntimeError('Bounded timing budget exceeded')
        if i % 2:
            b, a = timed(candidate), timed(native)
        else:
            a, b = timed(native), timed(candidate)
        pairs.append({'B0': a, 'P1': b, 'order': 'BA' if i % 2 else 'AB'})
    summaries = {}
    # 1 us is a conservative screen floor, not a measured timer specification.
    for metric in ('device_ms', 'wall_ms'):
        aa_max = max(abs(a[metric] - b[metric]) for a, b in aa)
        savings = [p['B0'][metric] - p['P1'][metric] for p in pairs]
        med = statistics.median(savings)
        summaries[metric] = {'native_median': statistics.median(p['B0'][metric] for p in pairs),
            'candidate_median': statistics.median(p['P1'][metric] for p in pairs),
            'paired_saving_median': med, 'AA_max_abs': aa_max, 'positive_pairs': sum(x > 0 for x in savings),
            'screen_passed': sum(x > 0 for x in savings) >= 9 and med > max(aa_max, 0.001)}
    result['timing'] = {'scope': 'Two-branch complete production manager decode, including input copy, guards, output clone and synchronization; excludes prefill/graph setup/HTTP/VAE',
                        'AA': aa, 'AB': pairs, 'summary': summaries, 'warmups_per_arm': 5}
    for step, branch, kw in inputs:
        expected = native.try_decode(**kw)
        check(f'post_timing_s{step}_{branch}_exact', bits(expected, candidate.try_decode(**kw)))
    check('timing_did_not_recapture', all(e.captures == 1 for m in (native, candidate) for e in m.entries.values()))
    save()
