# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HIP launch-correlated stage attribution; keep unknown kernels explicit."""

from collections import Counter, defaultdict
import json
from pathlib import Path


def union_us(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    start, end = intervals[0]
    for a, b in intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


def family(name):
    if 'ck_tile::' in name and 'Fmha' in name:
        return 'attention_CK'
    if 'fmha_fwd' in name:
        return 'attention_ASM'
    if name.startswith('Cijk_') or 'gemm' in name.lower() or 'wvSplitK' in name:
        return 'GEMM'
    if 'native_layer_norm' in name:
        return 'layer_norm_and_modulation'
    if 'rsqrt' in name and ('view_as_complex' in name or 'split_with_sizes' in name):
        return 'QK_norm_and_rope_preparation'
    if 'complex<float>' in name or 'rotary' in name.lower():
        return 'rotary'
    if 'cat' in name and name.startswith('triton_'):
        return 'concat_and_fused_copies'
    if 'silu' in name:
        return 'activation'
    if 'residual' in name or name.startswith('triton_poi_fused_add_mul'):
        return 'residual_and_modulation'
    if 'copy' in name.lower() or 'memcpy' in name.lower() or 'clone' in name:
        return 'copy_and_layout'
    if 'zero' in name.lower() or 'fill' in name.lower() or 'memset' in name.lower():
        return 'initialization'
    return 'other'


def analyze(trace_path, output_path):
    events = json.loads(Path(trace_path).read_text())['traceEvents']
    complete = [x for x in events if x.get('ph') == 'X']
    ranges = [x for x in complete if x.get('cat') == 'user_annotation' and x['name'].startswith('qwen21/')]
    phases = [x for x in ranges if x['name'].split('/')[1] in ('prefill', 'capture_decode', 'steady_decode')]
    blocks = [x for x in ranges if x['name'].startswith('qwen21/block/')]
    runtime = {x.get('args', {}).get('correlation'): x for x in complete
               if x.get('cat') == 'cuda_runtime' and 'correlation' in x.get('args', {})}
    cpu = {x.get('args', {}).get('External id'): x for x in complete
           if x.get('cat') == 'cpu_op' and 'External id' in x.get('args', {})}
    device = [x for x in complete if x.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    grouped = defaultdict(list)
    named = defaultdict(list)
    mapped = Counter()
    block_gemms = defaultdict(list)

    def contains(a, b):
        return a.get('tid') == b.get('tid') and a['ts'] <= b['ts'] <= a['ts'] + a['dur']

    for e in device:
        args = e.get('args', {})
        host = runtime.get(args.get('correlation'))
        method = 'HIP_correlation'
        if host is None:
            host = cpu.get(args.get('External id'))
            method = 'external_id'
        owners = [] if host is None else [r for r in phases if contains(r, host)]
        if owners:
            owner = min(owners, key=lambda x: x['dur'])
            stage = owner['name'].split('/')[1]
            mapped[method] += 1
        else:
            stage = 'outside_or_unassigned'
            mapped['outside_or_unassigned'] += 1
        category = family(e['name']) if e['cat'] == 'kernel' else e['cat']
        item = dict(name=e['name'], ts=e['ts'], dur=e['dur'], category=category,
                    stream=args.get('stream'), grid=args.get('grid'), block=args.get('block'))
        grouped[stage].append(item)
        named[(stage, e['name'])].append(item)
        if category == 'GEMM' and host is not None and stage == 'prefill':
            bb = [r for r in blocks if contains(r, host)]
            if bb:
                block = min(bb, key=lambda x: x['dur'])
                block_gemms[(owner['name'], block['name'], block['ts'])].append(item)

    output = dict(mapping=dict(mapped), stages={},
                  sdpa_cpu_events=sum('scaled_dot_product' in x['name'] for x in complete if x.get('cat') == 'cpu_op'),
                  annotations=[dict(name=x['name'], duration_ms=x['dur'] / 1000)
                               for x in ranges if not x['name'].startswith('qwen21/block/')],
                  launch_api_counts=dict(Counter(x['name'] for x in runtime.values())))
    for stage, items in grouped.items():
        groups = defaultdict(list)
        for item in items:
            groups[item['category']].append(item)
        intervals = [(x['ts'], x['ts'] + x['dur']) for x in items]
        span = max(b for a, b in intervals) - min(a for a, b in intervals)
        busy = union_us(intervals)
        output['stages'][stage] = dict(
            device_events=len(items), device_sum_ms=sum(x['dur'] for x in items) / 1000,
            device_busy_union_ms=busy / 1000, device_span_ms=span / 1000,
            no_recorded_device_work_ms=(span - busy) / 1000,
            groups={name: dict(count=len(vals), sum_ms=sum(x['dur'] for x in vals) / 1000)
                    for name, vals in sorted(groups.items(), key=lambda pair: -sum(x['dur'] for x in pair[1]))},
            top_kernels=[dict(name=name, count=len(vals), sum_ms=sum(x['dur'] for x in vals) / 1000,
                              grids=[list(g) for g in sorted({tuple(x['grid']) for x in vals if x['grid']})])
                         for (s, name), vals in sorted(named.items(), key=lambda pair: -sum(x['dur'] for x in pair[1]))
                         if s == stage][:24],
        )
    # Source order: QKV projection, attention output projection, FFN input, FFN output.
    # Only assign blocks with exactly four GEMM kernel events; leave others unclassified.
    order_groups = defaultdict(list)
    for vals in block_gemms.values():
        if len(vals) == 4:
            for name, val in zip(('QKV_projection', 'attention_output_projection', 'FFN_input', 'FFN_output'),
                                 sorted(vals, key=lambda x: x['ts']), strict=True):
                order_groups[name].append(val['dur'])
    output['prefill_GEMM_order_mapping'] = dict(
        block_calls=len(block_gemms), four_GEMM_blocks=sum(len(v) == 4 for v in block_gemms.values()),
        groups={k: dict(count=len(v), sum_ms=sum(v) / 1000) for k, v in order_groups.items()},
        note='Order mapping requires exactly four GEMM events per source block; not an instruction-level profile.')
    Path(output_path).write_text(json.dumps(output, indent=2) + '\n')
    return output
