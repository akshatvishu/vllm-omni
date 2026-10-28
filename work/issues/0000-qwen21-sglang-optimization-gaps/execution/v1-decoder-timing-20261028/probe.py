"""Bounded eager decoder timing; keep convolution-policy and V1 effects separate."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def summary(rows):
    return {metric: dict(median=statistics.median(r[metric] for r in rows),
                         minimum=min(r[metric] for r in rows),
                         maximum=max(r[metric] for r in rows))
            for metric in ('wall_ms', 'event_ms')}


def worker(args):
    inherited = {k: os.environ.get(k) for k in
                 ('MIOPEN_DEBUG_CONV_DIRECT', 'MIOPEN_DEBUG_CONV_GEMM')}
    assert inherited == {'MIOPEN_DEBUG_CONV_DIRECT': '0', 'MIOPEN_DEBUG_CONV_GEMM': '0'}, inherited
    if args.policy == 'deterministic':
        for key in inherited:
            os.environ.pop(key)
    # Logging is disabled before importing Torch/MIOpen in every timed worker.
    os.environ.update(MIOPEN_ENABLE_LOGGING='0', MIOPEN_ENABLE_LOGGING_CMD='0', MIOPEN_LOG_LEVEL='1')
    import torch
    from vllm_omni.diffusion.models.qwen_image_21 import vae_experiment as exp
    from vllm_omni.diffusion.models.qwen_image_21.autoencoder_kl_qwenimage21 import (
        AutoencoderKLQwenImage21, _cudnn_deterministic)
    from vllm_omni.diffusion.layers import vae_norm_epilogue, vae_rms_norm

    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', policy=args.policy, warmups=[], samples=[], controls=[], postchecks=[])
    started = time.monotonic()

    def save():
        write(args.output / 'findings.json', result)

    def meta(t):
        return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                    offset=t.storage_offset(), device=str(t.device))

    def cpu_bytes(t):
        return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

    try:
        torch.cuda.set_device(0)
        arch = torch.cuda.get_device_properties(0).gcnArchName
        assert torch.version.hip and 'gfx942' in arch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = args.policy == 'deterministic'
        assert torch.backends.cudnn.enabled and torch.backends.cudnn.allow_tf32
        assert not torch.are_deterministic_algorithms_enabled()
        assert digest(args.input) == '490010cd207fcde98634a4455c24181d4b84b97e37254070ab6f7a2ac0141142'
        saved = torch.load(args.input, map_location='cpu', weights_only=True)
        assert saved['first_chunk'] is True and saved['input'].dtype == torch.bfloat16
        x = saved['input'].cuda()
        frozen_input = cpu_bytes(x.cpu())
        vae = AutoencoderKLQwenImage21.from_pretrained(str(args.checkpoint), subfolder='vae',
            local_files_only=True, torch_dtype=torch.bfloat16).to('cuda').eval()
        keys = tuple(vae.state_dict())
        result.update(source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      input_sha256=digest(args.input), torch=torch.__version__, torch_git=torch.version.git_version,
                      hip=torch.version.hip, arch=arch, inherited_exclusions=inherited,
                      miopen_env={k: v for k, v in os.environ.items() if k.startswith('MIOPEN')},
                      backend_flags={k: getattr(torch.backends.cudnn, k) for k in
                                     ('enabled', 'benchmark', 'deterministic', 'allow_tf32')},
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      libraries={m.__name__: digest(inspect.getfile(m)) for m in
                                 (vae_norm_epilogue, vae_rms_norm, exp)},
                      timing_boundary='synchronized eager vae.decoder(x, first_chunk=True); includes dispatch and allocations; excludes output checks, copies, profiling and solver logging',
                      event_meaning='HIP event interval includes host submission gaps; not summed kernel time')
        reference_bytes = None
        reference_meta = None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        def invoke(arm, label, timed=False, verify=False):
            nonlocal reference_bytes, reference_meta
            exp.DISABLED = arm == 'native'
            exp.VERIFY_EVERY_CALL = verify
            before = Counter(exp.COUNTS)
            torch.cuda.synchronize()
            if timed:
                start_event.record()
                t0 = time.perf_counter_ns()
            with _cudnn_deterministic():
                y = vae.decoder(x, first_chunk=True)
            if timed:
                end_event.record()
                end_event.synchronize()
                wall_ms = (time.perf_counter_ns() - t0) / 1e6
                event_ms = start_event.elapsed_time(end_event)
            else:
                torch.cuda.synchronize()
            # All validation and host copies occur after the measured interval.
            counts = dict(Counter(exp.COUNTS) - before)
            cpu = y.detach().cpu()
            raw = cpu_bytes(cpu)
            metadata = meta(y)
            finite = bool(torch.isfinite(cpu).all())
            if reference_bytes is None:
                reference_bytes, reference_meta = raw, metadata
                result['reference_sha256'] = hashlib.sha256(raw).hexdigest()
            row = dict(arm=arm, label=label, counts=counts, finite=finite,
                       exact=raw == reference_bytes, metadata_equal=metadata == reference_meta,
                       output_sha256=hashlib.sha256(raw).hexdigest(), output_metadata=metadata)
            if timed:
                row.update(wall_ms=wall_ms, event_ms=event_ms)
            category = 'samples' if timed and not label.startswith('aa-') else 'controls' if timed else 'postchecks' if label.startswith('post-') else 'warmups'
            result[category].append(row)
            save()
            assert finite and row['metadata_equal'], row
            if args.policy == 'deterministic':
                assert row['exact'], ('deterministic output differs', row)
            expected = {} if arm == 'native' else {'norm_silu/checked': 35, 'norm_silu/fused': 35} if verify else {'norm_silu/fused': 35}
            assert counts == expected, ('unexpected fusion dispatch or verification', row)
            return row

        with torch.inference_mode():
            result['silu_gate'] = vae_norm_epilogue.silu_epilogue_is_exact(x.device, x.dtype)
            assert result['silu_gate']
            invoke('native', 'warm-native-0')
            invoke('native', 'warm-native-1')
            if args.policy == 'deterministic':
                # Populate the actual candidate's two successful qualification entries.
                invoke('candidate', 'warm-candidate-0', verify=True)
                invoke('candidate', 'warm-candidate-1', verify=True)
                invoke('native', 'warm-native-steady')
                invoke('candidate', 'warm-candidate-steady')
                for i in range(2):
                    invoke('native', f'aa-before-{i}', timed=True)
                for pair in range(8):
                    order = ('native', 'candidate') if pair % 2 == 0 else ('candidate', 'native')
                    for arm in order:
                        invoke(arm, f'pair-{pair}', timed=True)
                for i in range(2):
                    invoke('native', f'aa-after-{i}', timed=True)
                invoke('native', 'post-native')
                invoke('candidate', 'post-candidate', verify=True)
            else:
                for i in range(8):
                    invoke('native', f'sample-{i}', timed=True)
                invoke('native', 'post-native')
            result['input_unchanged'] = cpu_bytes(x.cpu()) == frozen_input
            result['state_keys_preserved'] = tuple(vae.state_dict()) == keys
            assert result['input_unchanged'] and result['state_keys_preserved']
        result['summaries'] = {arm: summary([r for r in result['samples'] if r['arm'] == arm])
                               for arm in sorted({r['arm'] for r in result['samples']})}
        result['status'] = 'completed'
    except Exception:
        result.update(status='error', error=traceback.format_exc())
        print(result['error'], flush=True)
    finally:
        exp.DISABLED, exp.VERIFY_EVERY_CALL = False, False
        result['worker_wall_seconds'] = time.monotonic() - started
        save()
    print(json.dumps({k: result[k] for k in ('policy', 'status', 'worker_wall_seconds')}), flush=True)
    return int(result['status'] != 'completed')


def parent(args):
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', workers={}, paired={}, scope='one saved eager decoder input, exploratory timing only')
    prior = Path('/workspace/outputs/qwen21-v1-decoder-deterministic-20261028/result/findings.json')
    assert digest(prior) == 'e89a38f482396c88abc52347a72be09029b8c2a46962ceef93e6870830e3569e'
    result['prior_diagnostic_sha256'] = digest(prior)
    try:
        for name, policy in (('original-before', 'original'), ('deterministic', 'deterministic'), ('original-after', 'original')):
            command = [sys.executable, str(Path(__file__).resolve()), '--policy', policy,
                       '--checkpoint', str(args.checkpoint), '--input', str(args.input), '--output', str(args.output / name)]
            print('Starting worker ' + name, flush=True)
            with (args.output / (name + '.log')).open('x') as stream:
                run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=95)
            finding = args.output / name / 'findings.json'
            data = json.loads(finding.read_text())
            result['workers'][name] = dict(status=data['status'], sha256=digest(finding))
            write(args.output / 'findings.json', result)
            assert run.returncode == 0 and data['status'] == 'completed', (name, data.get('error'))
        data = {name: json.loads((args.output / name / 'findings.json').read_text()) for name in result['workers']}
        det = data['deterministic']
        old = [r for name in ('original-before', 'original-after') for r in data[name]['samples']]
        result['summaries'] = dict(original=summary(old), deterministic_native=det['summaries']['native'],
                                   deterministic_v1=det['summaries']['candidate'])
        for metric in ('wall_ms', 'event_ms'):
            paired = []
            for index in range(8):
                rows = {r['arm']: r[metric] for r in det['samples'] if r['label'] == f'pair-{index}'}
                paired.append(100 * (rows['native'] - rows['candidate']) / rows['native'])
            controls = det['controls']
            aa = [100 * abs(controls[i][metric] - controls[i + 1][metric]) / controls[i][metric] for i in (0, 2)]
            effect = statistics.median(paired)
            result['paired'][metric] = dict(reduction_pct=paired, median_reduction_pct=effect,
                minimum=min(paired), maximum=max(paired), positive_pairs=sum(v > 0 for v in paired),
                aa_abs_difference_pct=aa, exploratory_promising=effect > max(1.0, *aa) and sum(v > 0 for v in paired) >= 7)
            original = result['summaries']['original'][metric]['median']
            native = result['summaries']['deterministic_native'][metric]['median']
            fused = result['summaries']['deterministic_v1'][metric]['median']
            result['paired'][metric].update(policy_slowdown_pct=100 * (native / original - 1),
                deterministic_v1_vs_original_reduction_pct=100 * (1 - fused / original))
        result['matches_prior_instrumented_output'] = det['reference_sha256'] == json.loads(prior.read_text())['arms']['incumbent_a']['output_sha256']
        result['status'] = 'completed'
    except Exception:
        result.update(status='error', error=traceback.format_exc())
        print(result['error'], flush=True)
    write(args.output / 'findings.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return int(result['status'] != 'completed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', choices=('original', 'deterministic'))
    args = p.parse_args()
    raise SystemExit(worker(args) if args.policy else parent(args))
