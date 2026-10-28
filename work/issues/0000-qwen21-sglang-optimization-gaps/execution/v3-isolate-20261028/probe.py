"""Isolate V3 at actual upsample sites of one historical decoder input.

The diagnostic callback computes one biased main tensor and shares it between
the eager shortcut and Triton shortcut-add arms. It returns the native result
so candidate arithmetic never propagates to later sites. Test a changed source,
then convolution repetition, frozen no-bias addition and bias relocation.
No latency claim, model request or full decoder qualification is attempted.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import traceback

import torch
from torch.nn import functional as F


class GateFailure(Exception):
    pass


def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def compare(a, b):
    if a is None:
        return {'activated': False, 'exact': False}
    same_shape = a.shape == b.shape and a.dtype == b.dtype
    row = {'activated': True, 'same_shape_dtype': same_shape, 'exact': same_shape and bits(a, b),
           'same_stride': a.stride() == b.stride(), 'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
    if same_shape:
        af = a.contiguous().flatten()
        bf = b.contiguous().flatten()
        different = af.view(torch.int16) != bf.view(torch.int16)
        row.update(different_elements=int(different.sum()), max_abs=float((af.float() - bf.float()).abs().max()))
        if row['different_elements']:
            ix = different.nonzero()[:6, 0]
            row['examples'] = {'flat_indices': ix.tolist(), 'actual': af[ix].float().tolist(),
                               'expected': bf[ix].float().tolist(),
                               'actual_int16': af.view(torch.int16)[ix].tolist(),
                               'expected_int16': bf.view(torch.int16)[ix].tolist()}
    return row


def meta(t):
    return {'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype),
            'storage_offset': t.storage_offset()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'sites': [], 'scope': 'V3 boundary isolation; one historical decoder input; no performance measurement'}

    def save():
        (args.output / 'findings.json').write_text(json.dumps(result, indent=2) + '\n')

    def require(name, ok):
        if not ok:
            result['stop_reason'] = name
            save()
            raise GateFailure(name)

    try:
        from vllm_omni.diffusion.models.qwen_image_21.autoencoder_kl_qwenimage21 import (
            AutoencoderKLQwenImage21, _cudnn_deterministic,
        )
        from vllm_omni.diffusion.models.qwen_image_21 import vae_experiment as exp
        from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import triton_data_movement as primitive

        torch.cuda.set_device(0)
        arch = torch.cuda.get_device_properties(0).gcnArchName
        require('gfx942_required', 'gfx942' in arch and torch.version.hip is not None)
        input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
        require('historical_input_hash', input_hash == '490010cd207fcde98634a4455c24181d4b84b97e37254070ab6f7a2ac0141142')
        result.update(source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      input_sha256=input_hash, torch=torch.__version__, torch_git=torch.version.git_version,
                      hip=torch.version.hip, arch=arch,
                      primitive_sha256=hashlib.sha256(Path(inspect.getfile(primitive)).read_bytes()).hexdigest(),
                      cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        saved = torch.load(args.input, map_location='cpu', weights_only=True)
        require('saved_first_chunk_BF16', saved['first_chunk'] is True and saved['input'].dtype == torch.bfloat16)
        x = saved['input'].cuda()
        original_input = x.clone()
        vae = AutoencoderKLQwenImage21.from_pretrained(str(args.checkpoint), subfolder='vae',
            local_files_only=True, torch_dtype=torch.bfloat16).to('cuda').eval()
        names = {id(module): name for name, module in vae.named_modules()}
        old_helper = exp.bias_upsample
        state_keys = tuple(vae.state_dict())

        def diagnostic(owner, x, source, first_chunk):
            up, shortcut = owner.upsampler, owner.avg_shortcut
            row = {'site': names[id(owner)], 'source': meta(source), 'conv_input': meta(x),
                   'mode': up.mode, 'first_chunk': first_chunk,
                   'factor_t': shortcut.factor_t, 'factor_s': shortcut.factor_s, 'repeats': shortcut.repeats}
            result['sites'].append(row)
            before_x, before_source = x.clone(), source.clone()
            main = up(x)
            row['main'] = meta(main)
            frozen_main = main.clone()
            reference = main + shortcut(source, first_chunk=first_chunk)

            def fused(m, s, bias=None):
                return primitive.dup_up3d_add(m, s, shortcut.factor_t, shortcut.factor_s,
                    shortcut.repeats, first_chunk, main_bias=bias)

            row['shared_main'] = compare(fused(main, source), reference)
            # A controlled input change isolates gather/add; not a second image.
            changed = source + torch.tensor(0.125, dtype=source.dtype, device=source.device)
            row['changed_source_is_distinct'] = not bits(changed, source)
            row['changed_source_shared_main'] = compare(fused(main, changed), main + shortcut(changed, first_chunk=first_chunk))
            row['main_and_source_unchanged'] = bits(main, frozen_main) and bits(source, before_source) and bits(x, before_x)
            save()
            for name in ('shared_main', 'changed_source_shared_main'):
                if not all(row[name].get(k, False) for k in ('activated', 'exact', 'same_stride', 'finite')):
                    torch.save({'x': x.cpu(), 'source': source.cpu(), 'main': main.cpu(), 'first_chunk': first_chunk,
                                'site': row['site']}, args.output / 'failing-site.pt')
                    require(row['site'] + '/' + name, False)
            require(row['site'] + '/input_mutation', row['main_and_source_unchanged'])

            # Distinct question: can an unchanged biased upsampler repeat?
            biased_again = up(x)
            row['biased_upsampler_AA'] = compare(biased_again, main)
            b, c, t, h, w = x.shape
            flat = up.resample[0](x.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w))
            conv = up.resample[1]
            require('bounded_single_frame_no_time_cache', t == 1)

            def no_bias():
                y = F.conv2d(flat, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
                return y.view(b, t, *y.shape[1:]).permute(0, 2, 1, 3, 4)

            bare = no_bias()
            bare_again = no_bias()
            row['no_bias_conv_AA'] = compare(bare_again, bare)
            bias = conv.bias.to(bare.dtype)
            staged_main = bare + bias.view(1, -1, 1, 1, 1)
            row['frozen_bias_shortcut'] = compare(fused(bare, source, bias), staged_main + shortcut(source, first_chunk=first_chunk))
            row['biased_vs_deferred_main'] = compare(staged_main, main)
            row['biased_vs_deferred_output'] = compare(fused(bare, source, bias), reference)

            # Retain the old verifier's site identity without feeding its output
            # into subsequent sites. It still runs separate convolutions.
            exp.VERIFY_EVERY_CALL = True
            for add_only in (True, False):
                if hasattr(up, '_qwen_fusion_verdicts'):
                    up._qwen_fusion_verdicts.clear()
                exp.ADD_ONLY = add_only
                count = Counter(exp.COUNTS)
                observed = old_helper(owner, x, source, first_chunk)
                label = 'historical_add_only' if add_only else 'historical_full_fusion'
                row[label] = {'counts': dict(Counter(exp.COUNTS) - count),
                              'returned_vs_shared_reference': compare(observed, reference)}
            exp.VERIFY_EVERY_CALL = False
            exp.ADD_ONLY = False
            row['inputs_still_unchanged'] = bits(source, before_source) and bits(x, before_x) and bits(main, frozen_main)
            require(row['site'] + '/input_mutation_after_controls', row['inputs_still_unchanged'])
            print(json.dumps(row), flush=True)
            save()
            return reference

        exp.bias_upsample = diagnostic
        try:
            with torch.inference_mode(), _cudnn_deterministic():
                vae.decoder(x, first_chunk=True)
        finally:
            exp.bias_upsample = old_helper
            exp.VERIFY_EVERY_CALL = False
            exp.ADD_ONLY = False
        require('all_four_upsample_sites_observed', len(result['sites']) == 4)
        require('original_decoder_input_unchanged', bits(x, original_input))
        require('state_keys_preserved', state_keys == tuple(vae.state_dict()))
        result['status'] = 'diagnostic_completed'
    except GateFailure:
        result['status'] = 'stopped_at_failed_gate'
    except Exception:
        result['status'] = 'error'
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        save()
    return 1 if result['status'] == 'error' else 0


if __name__ == '__main__':
    raise SystemExit(main())
