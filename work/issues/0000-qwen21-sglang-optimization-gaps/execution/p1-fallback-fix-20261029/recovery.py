"""Full decode parity and bounded manager recovery after the fallback repair."""
import gc
import hashlib
import json
from pathlib import Path
from functools import wraps
from unittest.mock import patch
import weakref

import torch


def run(model, native, candidate, cache, registrations, latest_args,
        ctx, native_entry, check, result, save, output_dir):
    from vllm_omni.diffusion.models.qwen_image_21 import decode_graph as dg
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21KVCache

    report = result['recovery'] = dict(status='running', timing='not measured',
                                      comparisons={}, artifacts={}, injections={})
    tensors = output_dir / 'tensors'
    tensors.mkdir()
    managers = (('native', native), ('p1', candidate))
    native.max_entries = candidate.max_entries = 2

    def digest(t):
        return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def hashes(request):
        return {f'{i}/{branch}/{kind}': digest(row[branch][kind])
                for i, row in enumerate(request) for branch in ('cond', 'uncond') for kind in ('key', 'value')}

    def remember(label, t):
        path = tensors / (label + '.pt')
        torch.save(t.detach().cpu(), path)
        report['artifacts'][label] = dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                                         finite=bool(torch.isfinite(t).all()), content_sha256=digest(t),
                                         file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), path=str(path))
        save()

    def compare(label, expected, actual):
        if actual is None:
            check(label, False, error='unexpected manager fallback')
        remember(label, actual)
        same_contract = expected.shape == actual.shape and expected.dtype == actual.dtype
        row = dict(shape_equal=expected.shape == actual.shape, dtype_equal=expected.dtype == actual.dtype,
                   stride_equal=expected.stride() == actual.stride(),
                   both_finite=bool(torch.isfinite(expected).all()) and bool(torch.isfinite(actual).all()),
                   expected_sha256=digest(expected), actual_sha256=digest(actual))
        if same_contract:
            different = expected.contiguous().view(torch.uint8) != actual.contiguous().view(torch.uint8)
            row.update(byte_equal=not bool(different.any()), differing_bytes=int(different.sum()),
                       differing_elements=int(different.reshape(expected.numel(), expected.element_size()).any(1).sum()),
                       total_elements=expected.numel(), max_abs=float((expected.float() - actual.float()).abs().max()))
        report['comparisons'][label] = row
        check(label, same_contract and row['stride_equal'] and row['both_finite'] and row.get('byte_equal', False), **row)

    def key_for(manager, branch):
        keys = [key for key in manager.entries if key[0] == branch]
        assert len(keys) == 1, keys
        return keys[0]

    def state(manager, branch):
        entry = manager.entries[key_for(manager, branch)]
        ts = [*entry.k, *entry.v, *(entry.packed_k or []), *(entry.packed_v or [])]
        return dict(entry_ref=weakref.ref(entry), tensor_refs=[weakref.ref(t) for t in ts],
                    pointers=[t.data_ptr() for t in ts], captures=entry.captures)

    def released(snapshot):
        return snapshot['entry_ref']() is None and all(ref() is None for ref in snapshot['tensor_refs'])

    def register(manager, request, branch):
        meta = next(m for m in registrations if m['cache_branch'] == branch)
        if manager is native:
            with patch.object(dg, 'QwenImage21DecodeGraphEntry', native_entry):
                manager.register_prefill(kv_cache=request, **meta)
        else:
            manager.register_prefill(kv_cache=request, **meta)

    def kwargs(branch, request):
        return dict(latest_args[branch], kv_cache=request)

    def fallback(branch, request, manager=None):
        kw = kwargs(branch, request)
        kw['img_shapes'] = ctx['img_shapes']
        kw['encoder_hidden_states'] = ctx['negative_prompt_embeds' if branch == 'uncond' else 'prompt_embeds']
        with patch.object(model, '_decode_graph_manager', manager):
            return model(**kw, return_dict=False)[0]

    with torch.inference_mode(False), torch.no_grad():
        changed = QwenImage21KVCache([
            {branch: {kind: tensor.clone().add_(0.0625 if kind == 'key' else 0.125)
                      for kind, tensor in row[branch].items()} for branch in ('cond', 'uncond')}
            for row in cache])
    cache.clear()
    gc.collect()
    prior = json.loads(Path('/workspace/outputs/qwen21-p1-fallback-20261028/result/findings.json').read_text())
    before = hashes(changed)
    report['prefix_sha256_before'] = before
    check('saved_unconditional_prefixes_exact',
          {f'{i}/{kind}': before[f'{i}/uncond/{kind}'] for i in range(32) for kind in ('key', 'value')} ==
          prior['fallback']['prefix_sha256_before'])
    check('saved_decode_input_exact', digest(latest_args['uncond']['hidden_states']) ==
          prior['fallback']['tensors']['input_hidden']['sha256'])

    references = {}
    for branch in ('cond', 'uncond'):
        for name, manager in managers:
            register(manager, changed, branch)
            output = manager.try_decode(**kwargs(branch, changed))
            check(name + '_' + branch + '_graph_executes', output is not None)
            if name == 'native':
                references[branch] = output
                remember('native_graph_' + branch, output)
                compare('native_AA_' + branch, output, manager.try_decode(**kwargs(branch, changed)))
            else:
                compare('p1_graph_' + branch, references[branch], output)
            check(name + '_' + branch + '_buffer_policy',
                  (manager.entries[key_for(manager, branch)].packed_k is not None) == (name == 'p1'))
        compare('repaired_fallback_' + branch, references[branch], fallback(branch, changed))
    del output
    check('saved_graph_baseline_preserved', digest(references['uncond']) == prior['fallback']['tensors']['native_graph_1']['sha256'])

    # Observe the actual block arguments outside compilation. Preserve wrapper
    # metadata so the repair's compiled-callable check sees the same marker.
    originals = [block.forward for block in model.transformer_blocks]
    observed = []

    def observing(index):
        @wraps(originals[index])
        def call(**kw):
            observed.append((index, kw.get('prepared_modulation') is None))
            return originals[index](**kw)
        return call

    try:
        for index, block in enumerate(model.transformer_blocks):
            block.forward = observing(index)
        for branch in ('cond', 'uncond'):
            observed.clear()
            compare('observed_fallback_' + branch, references[branch], fallback(branch, changed))
            check('all_32_blocks_compute_modulation_' + branch,
                  observed == [(i, True) for i in range(32)])
    finally:
        for block, original in zip(model.transformer_blocks, originals):
            block.forward = original

    survivor_states = {name: state(manager, 'cond') for name, manager in managers}
    for name, manager in managers:
        # Test setup only: replace this probe's unconditional graph so the same
        # post-body failure is injected during a fresh capture. Keep cond alive.
        torch.cuda.synchronize()
        old_key = key_for(manager, 'uncond')
        manager.entries.pop(old_key)
        manager._owners.pop(old_key)
        gc.collect()
        register(manager, changed, 'uncond')
        failed_state = state(manager, 'uncond')
        failed_key = key_for(manager, 'uncond')
        injection = report['injections'][name] = dict(body_calls=0, captured_body_completed=False)
        original_body = model._decode_graph_forward

        def injected(entry):
            injection['body_calls'] += 1
            value = original_body(entry)
            if torch.cuda.is_current_stream_capturing():
                injection['captured_body_completed'] = True
                raise RuntimeError('P1 fallback repair test injected Python failure after captured decode body')
            return value

        with patch.object(model, '_decode_graph_forward', injected):
            check(name + '_failure_returns_none', manager.try_decode(**kwargs('uncond', changed)) is None)
            check(name + '_actual_capture_failure', injection['body_calls'] == 3 and injection['captured_body_completed'])
            check(name + '_failed_key_marker', manager.entries[failed_key] is None)
            check(name + '_no_capture_retry', manager.try_decode(**kwargs('uncond', changed)) is None and injection['body_calls'] == 3)
        torch.cuda.synchronize()
        gc.collect()
        check(name + '_failed_entry_buffers_released', released(failed_state))
        for repeat in (1, 2):
            compare(name + '_post_failure_fallback_' + str(repeat), references['uncond'], fallback('uncond', changed, manager))
        # Replay both survivors after each failure, including the manager whose
        # sibling graph used the same global graph pool and then failed capture.
        for survivor_name, survivor_manager in managers:
            compare(name + '_failure_survivor_' + survivor_name, references['cond'],
                    survivor_manager.try_decode(**kwargs('cond', changed)))
            current = state(survivor_manager, 'cond')
            previous = survivor_states[survivor_name]
            check(name + '_survivor_stable_' + survivor_name,
                  current['entry_ref']() is previous['entry_ref']() and current['pointers'] == previous['pointers']
                  and current['captures'] == previous['captures'])
    check('prefixes_unchanged_after_failures', hashes(changed) == before)

    # New request owner, same geometry, with changed first-layer values. Clone
    # first K in each branch because owner identity is that tensor's weakref.
    with torch.inference_mode(False), torch.no_grad():
        recovery = QwenImage21KVCache([{b: dict(values) for b, values in row.items()} for row in changed])
        for branch in ('cond', 'uncond'):
            recovery[0][branch]['key'] = recovery[0][branch]['key'].clone()
            recovery[0][branch]['value'] = recovery[0][branch]['value'] + 0.25
    owner_refs = [weakref.ref(changed[0][b]['key']) for b in ('cond', 'uncond')]
    changed.clear()
    torch.cuda.synchronize()
    gc.collect()
    check('old_request_owners_released', all(ref() is None for ref in owner_refs))
    for name, manager in managers:
        manager.release_idle_entries()
        check(name + '_idle_cleanup_clears_failure_and_survivor', not manager.entries and not manager._owners)
    gc.collect()
    check('survivor_entries_and_buffers_released', all(released(s) for s in survivor_states.values()))
    recovery_hashes = hashes(recovery)
    for branch in ('cond', 'uncond'):
        recovery_reference = None
        for name, manager in managers:
            register(manager, recovery, branch)
            output = manager.try_decode(**kwargs(branch, recovery))
            check(name + '_' + branch + '_recovery_executes', output is not None)
            if name == 'native':
                recovery_reference = output
                remember('recovery_native_' + branch, output)
                check('new_request_changes_output_' + branch, digest(output) != digest(references[branch]))
            else:
                compare('recovery_p1_' + branch, recovery_reference, output)
            compare('recovery_repeat_' + name + '_' + branch, recovery_reference,
                    manager.try_decode(**kwargs(branch, recovery)))
            check(name + '_' + branch + '_fresh_single_capture', manager.entries[key_for(manager, branch)].captures == 1)
        compare('recovery_fallback_' + branch, recovery_reference, fallback(branch, recovery))
    del output, recovery_reference
    check('new_request_prefixes_unchanged', hashes(recovery) == recovery_hashes)
    final_states = [state(manager, b) for _, manager in managers for b in ('cond', 'uncond')]
    recovery.clear()
    torch.cuda.synchronize()
    gc.collect()
    for name, manager in managers:
        manager.release_idle_entries()
        check(name + '_final_idle_cleanup', not manager.entries and not manager._owners)
    gc.collect()
    check('all_final_entry_buffers_released', all(released(s) for s in final_states))
    from torch._dynamo.utils import counters
    report['compiler_counters'] = {group: {str(k): int(v) for k, v in counts.items()} for group, counts in counters.items()}
    report['status'] = 'passed'
    save()
