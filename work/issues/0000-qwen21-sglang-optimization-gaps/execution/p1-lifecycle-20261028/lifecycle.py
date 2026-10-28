"""Untimed lifecycle checks on actual 32-block native and retained-prefix managers."""
import gc
import hashlib
from unittest.mock import patch
import weakref

import torch


def run(model, native, candidate, cache, registrations, latest_args, original_outputs,
        ctx, native_entry, check, bits, result, save):
    from vllm_omni.diffusion.models.qwen_image_21 import decode_graph as dg
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21KVCache

    result['lifecycle'] = dict(status='running', input_kind='controlled changes to captured prefixes; not new prompt embeddings',
                               timing='not measured', cases=[])
    native.max_entries = candidate.max_entries = 2
    managers = (('native', native), ('p1', candidate))

    def key_for(manager, branch, length=None):
        keys = [key for key in manager.entries if key[0] == branch and (length is None or key[2] == length)]
        assert len(keys) == 1, keys
        return keys[0]

    def state(manager, key):
        e = manager.entries[key]
        tensors = [*e.k, *e.v, *(e.packed_k or []), *(e.packed_v or [])]
        return dict(entry_ref=weakref.ref(e), tensor_refs=[weakref.ref(t) for t in tensors],
                    pointers=[t.data_ptr() for t in tensors], captures=e.captures)

    def released(snapshot):
        return snapshot['entry_ref']() is None and all(ref() is None for ref in snapshot['tensor_refs'])

    def same_storage(manager, key, snapshot):
        current = state(manager, key)
        return (current['entry_ref']() is snapshot['entry_ref']() and
                current['pointers'] == snapshot['pointers'] and current['captures'] == snapshot['captures'])

    def register(manager, request, metadata):
        if manager is native:
            with patch.object(dg, 'QwenImage21DecodeGraphEntry', native_entry):
                manager.register_prefill(kv_cache=request, **metadata)
        else:
            manager.register_prefill(kv_cache=request, **metadata)

    def kwargs(branch, request):
        return dict(latest_args[branch], kv_cache=request)

    def compare(branch, request, label):
        expected = native.try_decode(**kwargs(branch, request))
        actual = candidate.try_decode(**kwargs(branch, request))
        check(label + '_graphs_execute', expected is not None and actual is not None)
        check(label + '_output_exact', bits(expected, actual) and expected.stride() == actual.stride())
        check(label + '_finite', bool(torch.isfinite(expected).all()) and bool(torch.isfinite(actual).all()))
        for name, manager in managers:
            e = manager.entries[key_for(manager, branch, request[0][branch]['key'].shape[1])]
            check(label + '_' + name + '_prefix_exact', all(
                bits(e.k[i], row[branch]['key']) and bits(e.v[i], row[branch]['value'])
                for i, row in enumerate(request)))
        return actual

    def hashes(request):
        # Sequential CPU copies keep host memory bounded to a single prefix tensor.
        return {(i, branch, kind): hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                for i, block in enumerate(request) for branch, values in block.items() for kind, t in values.items()}

    # Request B has new tensors, the same shapes and different K/V in every layer/branch.
    with torch.inference_mode(False), torch.no_grad():
        changed = QwenImage21KVCache([
            {branch: {kind: tensor.clone().add_(0.0625 if kind == 'key' else 0.125)
                      for kind, tensor in values.items()} for branch, values in block.items()}
            for block in cache])
    check('changed_all_prefix_contents', all(not bits(changed[i][b][k], block[b][k])
          for i, block in enumerate(cache) for b in ('cond', 'uncond') for k in ('key', 'value')))
    changed_hashes = hashes(changed)
    original_states = {name: {b: state(manager, key_for(manager, b)) for b in ('cond', 'uncond')}
                       for name, manager in managers}
    original_owners = [weakref.ref(cache[0][b]['key']) for b in ('cond', 'uncond')]
    for metadata in registrations:
        b = metadata['cache_branch']
        for name, manager in managers:
            register(manager, changed, metadata)
            check(name + '_' + b + '_changed_live_owner_refused', manager.try_decode(**kwargs(b, changed)) is None)
            replay = manager.try_decode(**kwargs(b, cache))
            check(name + '_' + b + '_original_owner_preserved', replay is not None and bits(replay, original_outputs[b]))
    del replay
    cache.clear()
    gc.collect()
    check('original_owner_released', all(ref() is None for ref in original_owners))
    changed_outputs = {}
    for metadata in registrations:
        b = metadata['cache_branch']
        for name, manager in managers:
            register(manager, changed, metadata)
        changed_outputs[b] = compare(b, changed, 'changed_' + b)
        check('changed_' + b + '_affects_output', not bits(changed_outputs[b], original_outputs[b]))
        second = compare(b, changed, 'changed_' + b + '_repeat')
        check('changed_' + b + '_repeat_exact', bits(second, changed_outputs[b]))
        for name, manager in managers:
            check(name + '_' + b + '_reuse_addresses_without_recapture',
                  same_storage(manager, key_for(manager, b), original_states[name][b]))
    del second
    result['lifecycle']['cases'].append('changed-prefix ownership and replay passed')
    save()

    # A third cache key with one fewer prefix row forces real capacity eviction.
    # This controlled cache is not a new prompt or new image-layout qualification.
    short = QwenImage21KVCache([{'uncond': {k: t[:, :-1] for k, t in row['uncond'].items()}} for row in changed])
    short_meta = dict(next(m for m in registrations if m['cache_branch'] == 'uncond'))
    short_meta['prefix_len'] -= 1
    full_uncond_len = changed[0]['uncond']['key'].shape[1]
    victims = {}
    for name, manager in managers:
        replay = manager.try_decode(**kwargs('cond', changed))
        check(name + '_lru_touch_cond', replay is not None and bits(replay, changed_outputs['cond']))
        victim_key = key_for(manager, 'uncond', full_uncond_len)
        victims[name] = (victim_key, state(manager, victim_key))
        register(manager, short, short_meta)
        check(name + '_lru_oldest_key_evicted', victim_key not in manager.entries and victim_key not in manager._owners)
        check(name + '_lru_capacity_bound', len(manager.entries) == len(manager._owners) == 2)
        check(name + '_evicted_owner_falls_back', manager.try_decode(**kwargs('uncond', changed)) is None)
    del replay
    torch.cuda.synchronize()
    gc.collect()
    for name, (key, snapshot) in victims.items():
        check(name + '_evicted_entry_tensors_released', released(snapshot))
    short_output = compare('uncond', short, 'short_prefix')
    check('short_prefix_repeat_exact', bits(compare('uncond', short, 'short_prefix_repeat'), short_output))
    check('survivor_after_eviction_exact', bits(compare('cond', changed, 'survivor'), changed_outputs['cond']))
    for name, manager in managers:
        check(name + '_survivor_addresses_and_capture_count',
              same_storage(manager, key_for(manager, 'cond'), original_states[name]['cond']))
    result['lifecycle']['cases'].append('LRU eviction, tensor release and surviving graph passed')
    save()

    # Restore the ordinary unconditional key; the shortened key is now least recent.
    full_meta = next(m for m in registrations if m['cache_branch'] == 'uncond')
    short_states = {name: state(manager, key_for(manager, 'uncond', short_meta['prefix_len']))
                    for name, manager in managers}
    for name, manager in managers:
        register(manager, changed, full_meta)
    torch.cuda.synchronize()
    gc.collect()
    for name, snapshot in short_states.items():
        check(name + '_short_entry_evicted_and_released', released(snapshot))
    control = native.try_decode(**kwargs('uncond', changed))
    check('native_recapture_before_failure_exact', control is not None and bits(control, changed_outputs['uncond']))
    failure_key = key_for(candidate, 'uncond', full_uncond_len)
    failed_state = state(candidate, failure_key)
    injection = dict(body_calls=0, captured_body_completed=False)
    original_body = model._decode_graph_forward

    def injected(entry):
        injection['body_calls'] += 1
        output = original_body(entry)
        if torch.cuda.is_current_stream_capturing():
            injection['captured_body_completed'] = True
            raise RuntimeError('P1 lifecycle injected Python failure after captured decode body')
        return output

    with patch.object(model, '_decode_graph_forward', injected):
        failed = candidate.try_decode(**kwargs('uncond', changed))
        check('capture_failure_returns_native_fallback_signal', failed is None)
        check('failure_injected_inside_actual_capture', injection['body_calls'] == 3 and injection['captured_body_completed'])
        check('failed_key_has_bounded_none_marker', failure_key in candidate.entries and candidate.entries[failure_key] is None)
        check('failed_key_does_not_retry_capture', candidate.try_decode(**kwargs('uncond', changed)) is None and injection['body_calls'] == 3)
    result['lifecycle']['injection'] = injection
    torch.cuda.synchronize()
    gc.collect()
    check('failed_capture_entry_and_packed_tensors_released', released(failed_state))
    # Exercise the actual transformer forward fallback, not only the None signal.
    forward_args = kwargs('uncond', changed)
    forward_args['img_shapes'] = ctx['img_shapes']
    forward_args['encoder_hidden_states'] = ctx['negative_prompt_embeds']
    with patch.object(model, '_decode_graph_manager', candidate):
        fallback = model(**forward_args, return_dict=False)[0]
    check('actual_eager_fallback_exact', bits(fallback, changed_outputs['uncond']) and fallback.stride() == changed_outputs['uncond'].stride())
    check('survivor_after_capture_failure_exact', bits(compare('cond', changed, 'post_failure_survivor'), changed_outputs['cond']))
    check('changed_request_prefixes_unchanged', hashes(changed) == changed_hashes)
    result['lifecycle']['cases'].append('capture failure, buffer release and eager fallback passed')
    save()

    # A live owner may retain its weak owner record and None marker. After its
    # request is released, idle cleanup must remove both, permitting recapture.
    with torch.inference_mode(False), torch.no_grad():
        recovery = QwenImage21KVCache([{b: dict(values) for b, values in row.items()} for row in changed])
        for b in ('cond', 'uncond'):
            recovery[0][b]['key'] = recovery[0][b]['key'].clone()
            recovery[0][b]['value'] = recovery[0][b]['value'] + 0.25
    live_states = [state(manager, key) for _, manager in managers for key, entry in manager.entries.items() if entry is not None]
    short.clear()
    changed.clear()
    torch.cuda.synchronize()
    gc.collect()
    for name, manager in managers:
        manager.release_idle_entries()
        check(name + '_failed_owner_idle_cleanup', not manager.entries and not manager._owners)
    gc.collect()
    check('all_pre_recovery_entries_released', all(released(s) for s in live_states))
    for metadata in registrations:
        for name, manager in managers:
            register(manager, recovery, metadata)
        b = metadata['cache_branch']
        out = compare(b, recovery, 'recovery_' + b)
        check('recovery_' + b + '_new_contents_affect_output', not bits(out, changed_outputs[b]))
        for name, manager in managers:
            check(name + '_' + b + '_fresh_capture_after_failure', manager.entries[key_for(manager, b)].captures == 1)
        check('recovery_' + b + '_stable_replay', bits(compare(b, recovery, 'recovery_repeat_' + b), out))
    final_states = [state(manager, key) for _, manager in managers for key in manager.entries]
    recovery.clear()
    torch.cuda.synchronize()
    gc.collect()
    for name, manager in managers:
        manager.release_idle_entries()
        check(name + '_final_idle_cleanup', not manager.entries and not manager._owners)
    gc.collect()
    check('all_final_entries_and_buffers_released', all(released(s) for s in final_states))
    result['lifecycle']['cases'].append('new owner recapture and final release passed')
    result['lifecycle']['status'] = 'passed'
    result['lifecycle']['memory_scope'] = 'Weak-reference release and bounded manager state; no process peak or reserved-memory claim.'
    save()
