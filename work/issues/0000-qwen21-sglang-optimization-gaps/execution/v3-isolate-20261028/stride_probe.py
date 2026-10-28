"""Confirm the failed site's output strides using only its frozen operands.

No VAE execution or convolution. The optional view is a diagnostic correction,
not a production implementation or a relaxed acceptance rule.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from probe import bits, compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint')  # Controller-compatible; no weights are loaded.
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath.triton_data_movement import dup_up3d_add
    from vllm_omni.diffusion.models.qwen_image_21 import vae_experiment as exp

    data = torch.load(a.input, map_location='cpu', weights_only=True)
    original = json.loads(a.input.with_name('findings.json').read_text())
    row = original['sites'][-1]
    assert data['site'] == row['site'] == 'decoder.up_blocks.3'
    result = {'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'input_sha256': hashlib.sha256(a.input.read_bytes()).hexdigest(), 'cases': []}
    with torch.inference_mode():
        tensors = {name: torch.empty_strided(row[name]['shape'], row[name]['stride'], device='cuda', dtype=torch.bfloat16)
                   .copy_(data[name]) for name in ('main', 'source')}
        main, source = tensors['main'], tensors['source']
        b, c, t, h, w = source.shape
        ft, fs, rep = row['factor_t'], row['factor_s'], row['repeats']
        out_c = c * rep // (ft * fs * fs)
        for changed in (False, True):
            src = source + 0.125 if changed else source
            shortcut = src.repeat_interleave(rep, dim=1).view(b, out_c, ft, fs, fs, t, h, w)
            shortcut = shortcut.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous().view(b, out_c, t*ft, h*fs, w*fs)
            if data['first_chunk']:
                shortcut = shortcut[:, :, ft-1:]
            expected = main + shortcut
            actual = dup_up3d_add(main, src, ft, fs, rep, data['first_chunk'])
            assert actual is not None
            different_axes = [i for i, (x, y) in enumerate(zip(actual.stride(), expected.stride())) if x != y]
            assert different_axes and all(actual.shape[i] == 1 for i in different_axes)
            assert actual.is_contiguous() and expected.is_contiguous()
            corrected = actual.as_strided(expected.shape, expected.stride())
            r = {'changed_source': changed, 'main_stride': list(main.stride()),
                 'shortcut_stride': list(shortcut.stride()), 'actual_stride': list(actual.stride()),
                 'expected_stride': list(expected.stride()), 'different_axes': different_axes,
                 'original': compare(actual, expected), 'original_verifier_accepts': exp.same(actual, expected),
                 'contiguous_is_same_tensor': actual.contiguous() is actual,
                 'corrected_verifier_accepts': exp.same(corrected, expected),
                 'corrected_shares_storage': corrected.data_ptr() == actual.data_ptr(),
                 'view_preserves_bytes': bits(actual, corrected)}
            result['cases'].append(r)
        result['qualified_metadata_diagnosis'] = all(
            r['original']['exact'] and not r['original_verifier_accepts'] and r['corrected_verifier_accepts']
            and r['corrected_shares_storage'] and r['view_preserves_bytes'] for r in result['cases'])
    (a.output / 'findings.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['qualified_metadata_diagnosis'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
