"""Replay one captured Conv2d under default and deterministic MIOpen policies.

Fresh process per policy; no model load, norm fusion, solver override, timing
claim or request. One initial call and five unchanged-input repeats per arm.
MIOpen API/solver logs are retained for every call. One later profiled call
records kernel names only for a successfully executing policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

CAPTURE_SHA = '80a33e7d81330140d0d9448201695c55336903a34e968751727838e22e3ceb24'


def compare(a, b):
    import torch
    shape = a.shape == b.shape and a.dtype == b.dtype
    if not shape:
        return dict(exact=False, shape_dtype=False)
    av, bv = a.contiguous().reshape(-1), b.contiguous().reshape(-1)
    return dict(exact=torch.equal(av.view(torch.uint8), bv.view(torch.uint8)),
                shape_dtype=True, same_stride=a.stride() == b.stride(),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                different_elements=int((av.view(torch.int16) != bv.view(torch.int16)).sum()),
                max_abs=float((av.float() - bv.float()).abs().max()))


def worker(args):
    import torch
    import torch.nn.functional as F
    started = time.monotonic()
    out = args.output / args.policy
    out.mkdir(exist_ok=False)
    result = dict(policy=args.policy, status='running', repeats=[], timing='not measured')
    try:
        assert hashlib.sha256(args.capture.read_bytes()).hexdigest() == CAPTURE_SHA
        assert os.environ.get('MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC') is None
        torch.cuda.set_device(0)
        arch = torch.cuda.get_device_properties(0).gcnArchName
        assert torch.version.hip and 'gfx942' in arch
        assert not torch.are_deterministic_algorithms_enabled()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = args.policy == 'deterministic'
        torch.backends.cudnn.allow_tf32 = True
        capture = torch.load(args.capture, map_location='cpu', weights_only=True)
        assert capture['reference']['row']['op'] == 'aten.conv2d.default'
        assert len(capture['inputs']['leaves']) == 3
        assert capture['inputs']['schema'] == 'TreeSpec(tuple, None, [TreeSpec(tuple, None, [*,\n    *,\n    *]),\n  TreeSpec(dict, [], [])])'
        operands = [leaf['data'].to('cuda') for leaf in capture['inputs']['leaves']]
        for tensor, leaf in zip(operands, capture['inputs']['leaves'], strict=True):
            assert list(tensor.shape) == leaf['meta']['shape']
            assert list(tensor.stride()) == leaf['meta']['stride']
            assert tensor.storage_offset() == leaf['meta']['offset'] == 0
            assert tensor.dtype == torch.bfloat16
        x, weight, bias = operands
        result.update(torch=torch.__version__, torch_git=torch.version.git_version,
                      hip=torch.version.hip, arch=arch, capture_sha256=CAPTURE_SHA,
                      backend_flags=dict(enabled=torch.backends.cudnn.enabled,
                                         benchmark=torch.backends.cudnn.benchmark,
                                         deterministic=torch.backends.cudnn.deterministic,
                                         allow_tf32=torch.backends.cudnn.allow_tf32,
                                         deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
                      environment={k:v for k,v in os.environ.items() if k.startswith('MIOPEN_')},
                      operands=[dict(shape=list(t.shape), stride=list(t.stride()),
                                     pointer=t.data_ptr(), offset=t.storage_offset()) for t in operands])

        def run(label):
            print('QWEN_CONV_BEGIN ' + label, flush=True)
            # The saved aten.conv2d call had only these three arguments;
            # defaults are stride=1, padding=0, dilation=1 and groups=1.
            value = F.conv2d(x, weight, bias)
            torch.cuda.synchronize()
            print('QWEN_CONV_END ' + label, flush=True)
            return value

        with torch.inference_mode():
            try:
                first = run('first')
            except RuntimeError:
                result.update(status='first_call_failed', error=traceback.format_exc())
                return 0
            assert first.shape == (1, 1152, 32, 32)
            first_cpu = first.cpu()
            torch.save(first_cpu, out / 'first-output.pt')
            result['first_vs_historical'] = compare(first_cpu, capture['output']['leaves'][0]['data'])
            for i in range(5):
                value = run('repeat-' + str(i))
                row = compare(value.cpu(), first_cpu)
                row['index'] = i
                result['repeats'].append(row)
            result['input_unchanged'] = [compare(t.cpu(), leaf['data']) for t, leaf in
                                          zip(operands, capture['inputs']['leaves'], strict=True)]
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                profiled = run('later-profiled-call')
            trace = out / 'later-replay-trace.json'
            prof.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            result['later_replay_kernels'] = [e['name'] for e in events if e.get('cat') == 'kernel']
            result['later_profiled_vs_first'] = compare(profiled.cpu(), first_cpu)
            result['status'] = 'completed'
            result['exact_repeats'] = sum(r['exact'] for r in result['repeats'])
    except Exception:
        result.update(status='harness_error', error=traceback.format_exc())
    finally:
        result['worker_wall_seconds'] = time.monotonic() - started
        (out / 'findings.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    return int(result['status'] == 'harness_error')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', choices=['default', 'deterministic'])
    args = p.parse_args()
    if args.policy:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', arms={}, scope='frozen Conv2d repeatability only; no model or speed qualification',
                  source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    for policy in ('default', 'deterministic'):
        command = [sys.executable, str(Path(__file__).resolve()), '--capture', str(args.capture),
                   '--output', str(args.output), '--policy', policy]
        env = dict(os.environ, MIOPEN_ENABLE_LOGGING='1', MIOPEN_ENABLE_LOGGING_CMD='1', MIOPEN_LOG_LEVEL='7')
        with (args.output / (policy + '-miopen.log')).open('x') as log:
            try:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90)
                path = args.output / policy / 'findings.json'
                result['arms'][policy] = json.loads(path.read_text()) if path.exists() else dict(status='worker_failed', returncode=completed.returncode)
            except subprocess.TimeoutExpired:
                result['arms'][policy] = dict(status='worker_timeout')
        (args.output / 'findings.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(policy=policy, status=result['arms'][policy]['status'],
                              exact_repeats=result['arms'][policy].get('exact_repeats'))), flush=True)
    if all(a['status'] == 'completed' for a in result['arms'].values()):
        import torch
        a = torch.load(args.output / 'default/first-output.pt', map_location='cpu', weights_only=True)
        b = torch.load(args.output / 'deterministic/first-output.pt', map_location='cpu', weights_only=True)
        result['cross_policy_first_output'] = compare(b, a)
    result['status'] = 'diagnostic_completed'
    (args.output / 'findings.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
