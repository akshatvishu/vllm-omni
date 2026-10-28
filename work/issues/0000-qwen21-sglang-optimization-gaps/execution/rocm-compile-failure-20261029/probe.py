"""Stop the saved production graph before its first out-of-storage copy."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback

import torch


class UnsafeCopy(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'checks': [], 'events': []}

    def save():
        (args.output / 'findings.json').write_text(json.dumps(result, indent=2) + '\n')

    def check(name, passed, **detail):
        row = dict(name=name, passed=bool(passed), **detail)
        result['checks'].append(row)
        print(json.dumps(row), flush=True)
        save()
        assert passed, name

    def metadata(t):
        return dict(shape=list(t.shape), stride=list(t.stride()),
                    offset=t.storage_offset(), storage_bytes=t.untyped_storage().nbytes(),
                    available_elements=t.untyped_storage().nbytes() // t.element_size() - t.storage_offset())

    try:
        torch.cuda.set_device(0)
        check('gpu_health', torch.ones(128, device='cuda').sum().item() == 128)
        result['stack'] = dict(torch=torch.__version__, hip=torch.version.hip)
        import aiter.ops.mha  # Registers unchanged installed AITER operators.
        import vllm.model_executor.layers.utils  # Registers the unchanged GEMM.
        result['schemas'] = {name: str(getattr(torch.ops.aiter, name).default._schema)
                             for name in ('mha_fwd', 'fmha_v3_fwd')}
        source = Path('/workspace/outputs/qwen21-rocm-stage-profile-20261029-r1/cache/inductor/vn/cvnxlklyvxmhyc7f7nz4owbqplgq7gldju2uuunde7z6uye5yjk6.py')
        raw = source.read_text()
        result['generated_source'] = str(source)
        result['generated_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
        # These two generated kernels have one linear load and one linear store.
        for name in ('triton_poi_fused_8', 'triton_poi_fused_11'):
            tree = ast.parse(raw)
            body = next(n.value.args[1].value for n in tree.body
                        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
            check(name + '_linear_copy', 'tl.load(in_ptr0 + (x0), xmask)' in body
                  and 'tl.store(out_ptr0 + (x0), tmp0, xmask)' in body)
        spec = importlib.util.spec_from_file_location('saved_production_graph', source)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class Guard:
            def __init__(self, name, kernel):
                self.name, self.kernel = name, kernel

            def run(self, *operands, **kwargs):
                torch.cuda.synchronize()
                if self.name in ('triton_poi_fused_8', 'triton_poi_fused_11'):
                    src, dst, count = operands
                    row = dict(kernel=self.name, count=int(count), source=metadata(src), destination=metadata(dst))
                    result['events'].append(row)
                    save()
                    if count > row['source']['available_elements'] or count > row['destination']['available_elements']:
                        result['blocked_copy'] = row
                        save()
                        raise UnsafeCopy(self.name)
                result['last_started'] = self.name
                save()
                output = self.kernel.run(*operands, **kwargs)
                torch.cuda.synchronize()
                result['last_completed'] = self.name
                save()
                return output

        for name, kernel in list(vars(mod).items()):
            if name.startswith('triton_') and hasattr(kernel, 'run'):
                setattr(mod, name, Guard(name, kernel))
        # Shapes/strides and scalar lengths are the original graph's examples.
        # Values are synthetic. No latency or model correctness claim is made.
        operands = mod.get_args()
        operands[4].fill_(1e-6)
        torch.cuda.synchronize()
        check('saved_geometry', operands[0] == 19383 and operands[9] == 8 and operands[10] == 4040)
        try:
            mod.call(operands)
        except UnsafeCopy:
            check('unsafe_copy_prevented', True, **result['blocked_copy'])
        else:
            check('unsafe_copy_prevented', False)
        torch.cuda.synchronize()
        check('gpu_health_after', torch.ones(128, device='cuda').sum().item() == 128)
        result['status'] = 'completed'
    except BaseException:
        result['status'] = 'failed'
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        save()
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
