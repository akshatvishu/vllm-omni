"""One deterministic V1 incumbent A/A/candidate with matched instrumentation.

No forward hooks, tolerance changes, timing, or full requests. All ordinary
ATen operations are observed except uninitialized allocations. Norm/SiLU,
standalone RMSNorm and SDPA are explicit boundaries so internal allocation
and verifier operations cannot misalign the two logical execution streams.
Each arm snapshots inputs and outputs to CPU at the same boundaries. This
changes synchronization/history, so nonreproduction cannot establish a fix.
Stop each comparison at its first differing boundary and replay that boundary
three times on the identical live operands. Skip the candidate if A/A fails.
This policy differs from the historical default-policy baseline; no claim
of unchanged algorithm numerics, full-image quality or latency is made.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

# Policy is local to this fresh diagnostic process and set before library loading.
REMOVED_EXCLUSIONS = {key: os.environ.pop(key, None) for key in
                      ('MIOPEN_DEBUG_CONV_DIRECT', 'MIOPEN_DEBUG_CONV_GEMM')}
os.environ['MIOPEN_ENABLE_LOGGING'] = '1'
os.environ['MIOPEN_ENABLE_LOGGING_CMD'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '7'

import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class StopBoundary(Exception):
    pass


def meta(t):
    return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                device=str(t.device), offset=t.storage_offset(), pointer=t.data_ptr())


def bits(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                        b.contiguous().reshape(-1).view(torch.uint8)))


def take(value):
    flat, spec = tree_flatten(value)
    return dict(schema=repr(spec), leaves=[
        dict(meta=meta(v), data=v.detach().to('cpu', copy=True)) if isinstance(v, torch.Tensor)
        else dict(value=repr(v)) for v in flat])


def compare(a, b):
    structural = a['schema'] == b['schema'] and len(a['leaves']) == len(b['leaves'])
    report = dict(structure=structural, exact=structural, metadata=structural, leaves=[])
    if not structural:
        return report
    for aa, bb in zip(a['leaves'], b['leaves'], strict=True):
        if 'data' in aa and 'data' in bb:
            equal = bits(aa['data'], bb['data'])
            layout = all(aa['meta'][k] == bb['meta'][k] for k in ('shape', 'dtype', 'stride', 'offset', 'device'))
            item = dict(exact=equal, metadata=layout, actual=aa['meta'], expected=bb['meta'])
            if not equal and aa['data'].shape == bb['data'].shape and aa['data'].dtype == bb['data'].dtype:
                av, bv = aa['data'], bb['data']
                item['different_bytes'] = int((av.contiguous().reshape(-1).view(torch.uint8) !=
                                               bv.contiguous().reshape(-1).view(torch.uint8)).sum())
                if av.is_floating_point() and av.numel():
                    item['max_abs'] = float((av.float() - bv.float()).abs().max())
            report['exact'] &= equal
            report['metadata'] &= layout
            report['leaves'].append(item)
        else:
            equal = aa == bb
            report['exact'] &= equal
            report['metadata'] &= equal
    return report


def passed(row):
    return row['structure'] and row['exact'] and row['metadata']


class Trace(TorchDispatchMode):
    def __init__(self, output, result, names):
        super().__init__()
        self.output, self.result, self.names = output, result, names
        self.suppressed = False
        self.stack = []
        self.reference = []
        self.host_bytes = 0
        self.rows = []

    @contextmanager
    def silence(self):
        previous = self.suppressed
        self.suppressed = True
        try:
            yield
        finally:
            self.suppressed = previous

    def start(self, label):
        self.label = label
        self.index = 0
        self.rows = []
        self.result['arms'][label] = dict(status='running')

    def save(self):
        (self.output / 'findings.json').write_text(json.dumps(self.result, indent=2) + '\n')

    def boundary(self, label, fn, args, kwargs):
        if self.suppressed:
            return fn(*args, **kwargs)
        with self.silence():
            before = take((args, kwargs))
            row = dict(index=self.index, op=label, module=self.stack[-1] if self.stack else '<functional>')
            try:
                value = fn(*args, **kwargs)
            except Exception:
                capture = self.output / (self.label + '-failed-boundary-inputs.pt')
                torch.save(before, capture)
                row.update(error=traceback.format_exc(), input_metadata=[
                    leaf.get('meta', leaf) for leaf in before['leaves']],
                    capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest())
                self.result['arms'][self.label].update(status='operation_error', boundary=row,
                                                      completed_boundaries=self.index)
                self.save()
                raise
            after = take(value)
            if self.label == 'incumbent_a':
                self.reference.append(dict(row=row, inputs=before, outputs=after))
                self.host_bytes += sum(v['data'].numel() * v['data'].element_size()
                    for snap in (before, after) for v in snap['leaves'] if 'data' in v)
                if self.host_bytes > 32 * 1024**3:
                    raise RuntimeError('32 GiB host capture budget reached')
            else:
                if self.index >= len(self.reference):
                    raise RuntimeError('candidate emitted extra boundary')
                ref = self.reference[self.index]
                row['same_operation'] = row['op'] == ref['row']['op'] and row['module'] == ref['row']['module']
                row['inputs'] = compare(before, ref['inputs'])
                row['outputs'] = compare(after, ref['outputs'])
                if not row['same_operation'] or not passed(row['inputs']) or not passed(row['outputs']):
                    row['replays'] = []
                    # Only replay a boundary with the exact same upstream values
                    # and metadata; do not attribute inherited differences to it.
                    row['live_inputs_unchanged'] = compare(take((args, kwargs)), before)
                    mutable = bool(getattr(getattr(fn, '_schema', None), 'is_mutable', False))
                    row['replay_skipped_for_mutation'] = mutable or not passed(row['live_inputs_unchanged'])
                    if row['same_operation'] and passed(row['inputs']) and not row['replay_skipped_for_mutation']:
                        for _ in range(3):
                            replay = take(fn(*args, **kwargs))
                            row['replays'].append(dict(vs_original=compare(replay, after),
                                                      vs_incumbent=compare(replay, ref['outputs'])))
                        row['inputs_after_replay'] = compare(take((args, kwargs)), before)
                        # One diagnostic trace identifies the actual device kernel.
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
                            fn(*args, **kwargs)
                            torch.cuda.synchronize()
                        trace_path = self.output / (self.label + '-boundary-trace.json')
                        prof.export_chrome_trace(str(trace_path))
                        events = json.loads(trace_path.read_text())['traceEvents']
                        row['later_replay_device_kernels'] = [dict(name=e['name'], duration_us=e.get('dur')) for e in events
                                                              if e.get('cat') == 'kernel']
                    capture = self.output / (self.label + '-first-boundary.pt')
                    torch.save(dict(reference=ref, inputs=before, output=after), capture)
                    row['capture_sha256'] = hashlib.sha256(capture.read_bytes()).hexdigest()
                    self.result['arms'][self.label].update(status='first_difference', boundary=row)
                    self.save()
                    raise StopBoundary(label)
            self.rows.append(row)
            self.index += 1
            return value

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.suppressed or str(func).startswith(('aten.empty', 'aten.new_empty', 'aten.resize_')):
            return func(*args, **kwargs)
        return self.boundary(str(func), func, args, kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', arms={}, timing='not measured', scope='one historical decoder input; instrumented eager V1 with deterministic convolution and no Direct/GEMM exclusions')
    started = time.monotonic()
    trace = None
    try:
        from vllm_omni.diffusion.models.qwen_image_21 import vae_experiment as exp
        from vllm_omni.diffusion.models.qwen_image_21.autoencoder_kl_qwenimage21 import (
            AutoencoderKLQwenImage21, QwenImage21RMS_norm, _cudnn_deterministic)
        from vllm_omni.diffusion.layers import vae_norm_epilogue, vae_rms_norm
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.enabled and torch.backends.cudnn.allow_tf32
        torch.cuda.set_device(0)
        arch = torch.cuda.get_device_properties(0).gcnArchName
        assert torch.version.hip and 'gfx942' in arch
        input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
        assert input_hash == '490010cd207fcde98634a4455c24181d4b84b97e37254070ab6f7a2ac0141142'
        saved = torch.load(args.input, map_location='cpu', weights_only=True)
        assert saved['first_chunk'] is True and saved['input'].dtype == torch.bfloat16
        result.update(source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      input_sha256=input_hash, torch=torch.__version__, torch_git=torch.version.git_version,
                      hip=torch.version.hip, arch=arch,
                      libraries={m.__name__: hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest()
                                 for m in (vae_norm_epilogue, vae_rms_norm, exp)},
                      removed_exclusions=REMOVED_EXCLUSIONS,
                      miopen_env={k: v for k, v in os.environ.items() if k.startswith('MIOPEN')},
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      backend_flags={k: getattr(torch.backends.cudnn, k) for k in ('enabled', 'benchmark', 'deterministic', 'allow_tf32')})
        x = saved['input'].cuda()
        vae = AutoencoderKLQwenImage21.from_pretrained(str(args.checkpoint), subfolder='vae',
            local_files_only=True, torch_dtype=torch.bfloat16).to('cuda').eval()
        initial_keys = tuple(vae.state_dict())
        names = {id(m): n for n, m in vae.named_modules()}
        trace = Trace(args.output, result, names)
        old_helper, old_norm = exp.norm_silu, QwenImage21RMS_norm.forward
        old_call, old_attention = torch.nn.Module._call_impl, F.scaled_dot_product_attention

        def call(module, *a, **kw):
            trace.stack.append(names.get(id(module), type(module).__name__))
            try:
                return old_call(module, *a, **kw)
            finally:
                trace.stack.pop()

        def helper(owner, norm, act, value):
            before_counts = Counter(exp.COUNTS)
            out = trace.boundary('norm_silu/' + names[id(norm)],
                lambda v: old_helper(owner, norm, act, v), (value,), {})
            delta = dict(Counter(exp.COUNTS) - before_counts)
            if not exp.DISABLED and delta != {'norm_silu/checked': 1, 'norm_silu/fused': 1}:
                trace.result['arms'][trace.label].update(status='fusion_verifier_rejected',
                    boundary=dict(index=trace.index - 1, module=names[id(norm)], counts=delta,
                                  note='Fallback output must not be treated as candidate parity.'))
                trace.save()
                raise StopBoundary('fusion verifier rejected')
            return out

        def norm(module, value):
            return trace.boundary('rms_norm/' + names[id(module)],
                                  lambda v: old_norm(module, v), (value,), {})

        def attention(*a, **kw):
            return trace.boundary('scaled_dot_product_attention', old_attention, a, kw)

        exp.norm_silu, QwenImage21RMS_norm.forward = helper, norm
        torch.nn.Module._call_impl, F.scaled_dot_product_attention = call, attention
        try:
            with torch.inference_mode():
                # Resolve the cached SiLU arithmetic gate once before all arms.
                result['silu_gate'] = vae_norm_epilogue.silu_epilogue_is_exact(x.device, x.dtype)
                assert result['silu_gate']
                frozen_x = take(x)
                for label, disabled in (('incumbent_a', True), ('incumbent_b', True), ('candidate', False)):
                    if label == 'candidate' and result['arms']['incumbent_b']['status'] != 'completed':
                        result['arms']['candidate'] = dict(status='not_run', reason='incumbent repeatability failed')
                        break
                    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
                    exp.DISABLED, exp.VERIFY_EVERY_CALL = disabled, True
                    for module in vae.modules():
                        if hasattr(module, '_qwen_fusion_verdicts'):
                            module._qwen_fusion_verdicts.clear()
                    counts = Counter(exp.COUNTS)
                    trace.start(label)
                    try:
                        with trace, _cudnn_deterministic():
                            output = vae.decoder(x, first_chunk=True)
                        torch.cuda.synchronize()
                        result['arms'][label].update(status='completed', output=meta(output),
                                                     output_finite=bool(torch.isfinite(output).all()),
                                                     output_sha256=hashlib.sha256(output.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
                        if label != 'incumbent_a':
                            assert trace.index == len(trace.reference)
                    except StopBoundary:
                        pass
                    result['arms'][label].update(completed_boundaries=trace.index,
                        counts=dict(Counter(exp.COUNTS) - counts), input_unchanged=passed(compare(take(x), frozen_x)))
                    (args.output / (label + '-boundaries.json')).write_text(json.dumps(trace.rows, indent=2) + '\n')
                    trace.save()
                    print(json.dumps(dict(arm=label, status=result['arms'][label]['status'],
                                          completed_boundaries=trace.index, counts=result['arms'][label]['counts'])), flush=True)
                result['state_keys_preserved'] = tuple(vae.state_dict()) == initial_keys
                candidate = result['arms']['candidate']
                if candidate['status'] == 'completed':
                    assert candidate['counts'] == {'norm_silu/checked': 35, 'norm_silu/fused': 35}, candidate['counts']
                    assert len(trace.reference) == 306
                    assert result['state_keys_preserved']
                    assert all(arm['input_unchanged'] and arm['output_finite']
                               for arm in result['arms'].values())
                    assert len({arm['output_sha256'] for arm in result['arms'].values()}) == 1
                result['host_snapshot_bytes'] = trace.host_bytes
                if result['arms']['incumbent_b']['status'] != 'completed':
                    result['status'] = 'incumbent_not_repeatable'
                elif candidate['status'] == 'completed':
                    result['status'] = 'one_input_matched_deterministic_v1_pass'
                elif candidate['status'] == 'fusion_verifier_rejected':
                    result['status'] = 'fusion_verifier_rejected'
                else:
                    boundary = candidate['boundary']
                    result['status'] = ('execution_alignment_difference' if not boundary['same_operation']
                                        else 'upstream_input_difference' if not passed(boundary['inputs'])
                                        else 'candidate_boundary_value_difference' if not boundary['outputs']['exact']
                                        else 'candidate_boundary_metadata_difference')
        finally:
            exp.norm_silu, QwenImage21RMS_norm.forward = old_helper, old_norm
            torch.nn.Module._call_impl, F.scaled_dot_product_attention = old_call, old_attention
            exp.DISABLED, exp.VERIFY_EVERY_CALL = False, False
    except Exception:
        result.update(status='error', error=traceback.format_exc())
        print(result['error'], flush=True)
    finally:
        result['diagnostic_wall_seconds'] = time.monotonic() - started
        (args.output / 'findings.json').write_text(json.dumps(result, indent=2) + '\n')
    return int(result['status'] == 'error')


if __name__ == '__main__':
    raise SystemExit(main())
