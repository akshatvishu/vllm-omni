"""Untimed matched fallback controls for one actual 32-block decode input."""
import gc
import hashlib
from unittest.mock import patch
import weakref

import torch


def run(model, native, candidate, cache, registrations, latest_args,
        ctx, native_entry, check, result, save, output_dir):
    from vllm_omni.diffusion.models.qwen_image_21 import decode_graph as dg
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21KVCache

    diagnostic = result['fallback'] = dict(
        status='running', branch='uncond', timing='not measured',
        input_kind='same controlled prefix additions as lifecycle probe',
        tensors={}, comparisons={}, injections={},
    )
    tensors = {}
    tensor_dir = output_dir / 'tensors'
    tensor_dir.mkdir()

    def remember(name, tensor):
        # Record original metadata before CPU materialization or clone changes it.
        metadata = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                        stride=list(tensor.stride()), device=str(tensor.device),
                        storage_offset=tensor.storage_offset(),
                        finite=bool(torch.isfinite(tensor).all()))
        cpu = tensor.detach().cpu().contiguous()
        metadata['sha256'] = hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
        path = tensor_dir / (name + '.pt')
        torch.save(cpu, path)
        metadata['path'] = str(path)
        metadata['file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        diagnostic['tensors'][name] = metadata
        tensors[name] = cpu
        save()

    def compare(label, left, right):
        x, y = tensors[left], tensors[right]
        mx, my = diagnostic['tensors'][left], diagnostic['tensors'][right]
        row = dict(left=left, right=right, shape_equal=x.shape == y.shape,
                   dtype_equal=x.dtype == y.dtype, stride_equal=mx['stride'] == my['stride'],
                   both_finite=mx['finite'] and my['finite'])
        if row['shape_equal'] and row['dtype_equal']:
            differing = x.view(torch.uint8) != y.view(torch.uint8)
            row.update(byte_equal=not bool(differing.any()),
                       differing_bytes=int(differing.sum()), total_bytes=x.numel() * x.element_size(),
                       differing_elements=int(differing.reshape(x.numel(), x.element_size()).any(dim=1).sum()),
                       total_elements=x.numel())
            if row['both_finite']:
                delta = x.float() - y.float()
                row.update(max_abs=float(delta.abs().max()), rmse=float(delta.square().mean().sqrt()))
        diagnostic['comparisons'][label] = row
        print('COMPARISON', label, row, flush=True)
        save()
        return row

    def prefix_hashes(request):
        return {f'{i}/{kind}': hashlib.sha256(
                    row['uncond'][kind].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest() for i, row in enumerate(request) for kind in ('key', 'value')}

    # Only the failing unconditional branch is exercised. Conditional prefill
    # was required to form the same CFG Euler update used in the previous test.
    with torch.inference_mode(False), torch.no_grad():
        changed = QwenImage21KVCache([
            {'uncond': {kind: tensor.clone().add_(0.0625 if kind == 'key' else 0.125)
                        for kind, tensor in row['uncond'].items()}}
            for row in cache
        ])
    cache.clear()
    gc.collect()
    kwargs = dict(latest_args['uncond'], kv_cache=changed)
    metadata = next(m for m in registrations if m['cache_branch'] == 'uncond')
    forward_args = dict(kwargs, img_shapes=ctx['img_shapes'], encoder_hidden_states=ctx['negative_prompt_embeds'])
    remember('input_hidden', kwargs['hidden_states'])
    remember('input_timestep', kwargs['timestep'])
    diagnostic['prefix_sha256_before'] = prefix_hashes(changed)
    diagnostic['geometry'] = dict(prefix=changed[0]['uncond']['key'].shape[1],
                                  target=metadata['target_freqs'].shape[0], layers=len(changed))
    save()

    def register(manager):
        if manager is native:
            with patch.object(dg, 'QwenImage21DecodeGraphEntry', native_entry):
                manager.register_prefill(kv_cache=changed, **metadata)
        else:
            manager.register_prefill(kv_cache=changed, **metadata)
        check(('native' if manager is native else 'p1') + '_one_entry', len(manager.entries) == 1)

    # Capture native and P1 graphs with the same changed prefix, without the
    # already validated ownership/eviction sequence. Save every output at once
    # because entries share the platform graph memory pool.
    for name, manager in (('native', native), ('p1', candidate)):
        register(manager)
        entry = next(iter(manager.entries.values()))
        check(name + '_buffer_policy', (entry.packed_k is not None) == (name == 'p1'))
        del entry
        for repeat in (1, 2):
            value = manager.try_decode(**kwargs)
            check(name + '_graph_executes_' + str(repeat), value is not None)
            remember(name + '_graph_' + str(repeat), value)
            del value
        compare(name + '_graph_AA', name + '_graph_1', name + '_graph_2')
    compare('native_vs_p1_graph', 'native_graph_1', 'p1_graph_1')

    # An empty manager returns None without injecting errors. The full model
    # executes its ordinary fallback path, keeping regional compilation active.
    bypass = dg.QwenImage21DecodeGraphManager(model)

    def forward_fallback(manager, label):
        with patch.object(model, '_decode_graph_manager', manager):
            remember(label, model(**forward_args, return_dict=False)[0])

    for repeat in (1, 2):
        forward_fallback(bypass, 'native_fallback_pre_' + str(repeat))
    compare('native_fallback_pre_AA', 'native_fallback_pre_1', 'native_fallback_pre_2')
    compare('native_graph_vs_fallback_before_failure', 'native_graph_1', 'native_fallback_pre_1')

    # Recreate only the selected manager entry to guarantee a new capture.
    # Use identical Python failure points for native and P1. A mismatch is an
    # observation, while failure to execute this protocol is a hard gate.
    for name, manager in (('native', native), ('p1', candidate)):
        manager.clear()
        torch.cuda.synchronize()
        gc.collect()
        register(manager)
        key = next(iter(manager.entries))
        entry = manager.entries[key]
        refs = [weakref.ref(entry), *(weakref.ref(t) for t in [
            *entry.k, *entry.v, *(entry.packed_k or []), *(entry.packed_v or [])])]
        del entry
        injection = diagnostic['injections'][name] = dict(body_calls=0, captured_body_completed=False)
        original_body = model._decode_graph_forward

        def injected(entry):
            injection['body_calls'] += 1
            value = original_body(entry)
            if torch.cuda.is_current_stream_capturing():
                injection['captured_body_completed'] = True
                raise RuntimeError('P1 fallback diagnostic injected Python failure after captured decode body')
            return value

        with patch.object(model, '_decode_graph_forward', injected):
            value = manager.try_decode(**kwargs)
            check(name + '_failure_returns_none', value is None)
            check(name + '_failure_point', injection['body_calls'] == 3 and injection['captured_body_completed'])
            check(name + '_failure_marker', key in manager.entries and manager.entries[key] is None)
            check(name + '_failure_does_not_retry', manager.try_decode(**kwargs) is None and injection['body_calls'] == 3)
        torch.cuda.synchronize()
        gc.collect()
        check(name + '_failed_entry_released', all(ref() is None for ref in refs))
        for repeat in (1, 2):
            label = name + '_fallback_post_' + str(repeat)
            forward_fallback(manager, label)
            compare(label + '_vs_pre', 'native_fallback_pre_1', label)
        compare(name + '_fallback_post_AA', name + '_fallback_post_1', name + '_fallback_post_2')
    compare('native_vs_p1_post_failure', 'native_fallback_post_1', 'p1_fallback_post_1')

    diagnostic['prefix_sha256_after'] = prefix_hashes(changed)
    check('request_prefixes_unchanged', diagnostic['prefix_sha256_before'] == diagnostic['prefix_sha256_after'])
    # This is explicit probe teardown, not a claim about automatic recovery.
    for manager in (native, candidate, bypass):
        manager.clear()
    changed.clear()
    torch.cuda.synchronize()
    gc.collect()
    check('probe_managers_explicitly_cleared', all(not m.entries and not m._owners for m in (native, candidate, bypass)))
    diagnostic['status'] = 'completed'
    save()
