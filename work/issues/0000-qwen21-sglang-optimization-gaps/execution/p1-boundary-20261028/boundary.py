"""Observe compiled-block boundaries, then vary the first differing block's arguments."""
from contextlib import contextmanager
import gc
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch


def run(model, native, candidate, cache, registrations, latest_args,
        ctx, native_entry, check, result, save, output_dir):
    from vllm_omni.diffusion.models.qwen_image_21 import decode_graph as dg
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21KVCache

    report = result['boundary'] = dict(status='running', comparisons={}, artifacts={},
                                      code_modules={}, inference_contexts={})
    tensors_dir = output_dir / 'tensors'
    tensors_dir.mkdir()

    def digest(t):
        return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def metadata(t):
        return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                    storage_offset=t.storage_offset(), inference=t.is_inference())

    def compare(label, x, y):
        row = dict(left=metadata(x), right=metadata(y), shape_equal=x.shape == y.shape,
                   dtype_equal=x.dtype == y.dtype, stride_equal=x.stride() == y.stride(),
                   both_finite=bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()))
        if row['shape_equal'] and row['dtype_equal']:
            xb, yb = x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)
            unequal = xb != yb
            row.update(byte_equal=not bool(unequal.any()), differing_bytes=int(unequal.sum()),
                       total_elements=x.numel(), differing_elements=int(
                           unequal.reshape(x.numel(), x.element_size()).any(dim=1).sum()))
            if row['both_finite']:
                delta = (x - y).abs() if x.is_complex() else (x.float() - y.float()).abs()
                row.update(max_abs=float(delta.max()), rmse=float(delta.float().square().mean().sqrt()))
        report['comparisons'][label] = row
        print('COMPARE', label, row, flush=True)
        save()
        return row.get('byte_equal', False)

    def artifact(name, tensor):
        path = tensors_dir / (name + '.pt')
        torch.save(tensor.detach().cpu(), path)
        report['artifacts'][name] = dict(**metadata(tensor), content_sha256=digest(tensor),
                                       file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), path=str(path))
        save()

    def modules(phase):
        # Inspection only; do not replace the compiler or mutate its cache.
        from torch._inductor.codecache import PyCodeCache
        rows = []
        dest = output_dir / 'generated'
        dest.mkdir(exist_ok=True)
        for module in PyCodeCache.modules:
            filename = getattr(module, '__file__', None)
            if filename and Path(filename).is_file():
                raw = Path(filename).read_bytes()
                target = dest / Path(filename).name
                if not target.exists():
                    target.write_bytes(raw)
                rows.append(dict(module=module.__name__, path=str(target), sha256=hashlib.sha256(raw).hexdigest()))
        report['code_modules'][phase] = rows
        from torch._dynamo.utils import counters
        report.setdefault('compiler_counters', {})[phase] = {
            group: {str(k): int(v) for k, v in values.items()} for group, values in counters.items()
        }
        save()

    # Match the previous focused diagnostic, including its controlled prefixes.
    with torch.inference_mode(False), torch.no_grad():
        changed = QwenImage21KVCache([
            {'uncond': {kind: tensor.clone().add_(0.0625 if kind == 'key' else 0.125)
                        for kind, tensor in row['uncond'].items()}}
            for row in cache])
    cache.clear()
    gc.collect()
    kwargs = dict(latest_args['uncond'], kv_cache=changed)
    registration = next(m for m in registrations if m['cache_branch'] == 'uncond')
    forward_args = dict(kwargs, img_shapes=ctx['img_shapes'], encoder_hidden_states=ctx['negative_prompt_embeds'])
    check('saved_decode_input_exact', digest(kwargs['hidden_states']) ==
          '13ab2100edb626c4d5e7b64014731eefb80155291194e18b872235497404af8a')
    previous = __import__('json').loads(Path(
        '/workspace/outputs/qwen21-p1-fallback-20261028/result/findings.json').read_text())
    hashes = {f'{i}/{kind}': digest(row['uncond'][kind])
              for i, row in enumerate(changed) for kind in ('key', 'value')}
    check('saved_prefixes_exact', hashes == previous['fallback']['prefix_sha256_before'])
    report['prefix_sha256_before'] = hashes

    def register():
        with patch.object(dg, 'QwenImage21DecodeGraphEntry', native_entry):
            native.register_prefill(kv_cache=changed, **registration)
        check('native_only_entry', len(native.entries) == 1 and
              next(iter(native.entries.values())).packed_k is None)

    def fallback():
        with patch.object(model, '_decode_graph_manager', None):
            return model(**forward_args, return_dict=False)[0]

    register()
    graph_ref = native.try_decode(**kwargs)
    check('native_graph_executes', graph_ref is not None)
    check('native_graph_matches_saved', digest(graph_ref) == previous['fallback']['tensors']['native_graph_1']['sha256'])
    check('native_graph_AA', compare('graph_AA', graph_ref, native.try_decode(**kwargs)))
    modules('native_graph')
    fallback_ref = fallback()
    check('native_fallback_matches_saved', digest(fallback_ref) == previous['fallback']['tensors']['native_fallback_pre_1']['sha256'])
    check('native_fallback_AA', compare('fallback_AA', fallback_ref, fallback()))
    check('known_difference_reproduced', not compare('graph_vs_fallback', graph_ref, fallback_ref))
    modules('native_fallback')
    artifact('graph_reference', graph_ref)
    artifact('fallback_reference', fallback_ref)

    originals = [block.forward for block in model.transformer_blocks]
    buffers = {'graph': {}, 'fallback': {}}
    arguments = {'graph': {}, 'fallback': {}}
    observed_metadata = {'graph': {}, 'fallback': {}}

    def copy(arm, label, tensor):
        observed_metadata[arm][label] = metadata(tensor)
        if label not in buffers[arm]:
            # Match inference status and strides. Allocate during warmup, never
            # for the first time inside graph capture.
            assert not torch.cuda.is_current_stream_capturing()
            with torch.inference_mode(tensor.is_inference()):
                buffers[arm][label] = torch.empty_strided(tensor.shape, tensor.stride(),
                                                         dtype=tensor.dtype, device=tensor.device)
        buffers[arm][label].copy_(tensor)

    @contextmanager
    def observe(arm):
        def wrapper(index):
            def call(**kw):
                arguments[arm][index] = dict(kw)
                report['inference_contexts'][arm] = torch.is_inference_mode_enabled()
                copy(arm, f'b{index}_input', kw['hidden_states'])
                if index == 0:
                    for field in ('modulation', 'freqs', 'target_token_mask'):
                        copy(arm, field, kw[field])
                    if kw.get('prepared_modulation') is not None:
                        for n, tensor in enumerate(kw['prepared_modulation']):
                            copy(arm, f'prepared_{n}', tensor)
                output = originals[index](**kw)
                copy(arm, f'b{index}_output', output)
                return output
            return call
        try:
            for index, block in enumerate(model.transformer_blocks):
                block.forward = wrapper(index)
            yield
        finally:
            for block, original in zip(model.transformer_blocks, originals):
                block.forward = original

    native.clear()
    torch.cuda.synchronize()
    gc.collect()
    register()
    with observe('graph'):
        graph_observed = native.try_decode(**kwargs)
    check('instrumented_graph_executes', graph_observed is not None)
    check('observation_preserves_graph_output', compare('observed_graph_control', graph_ref, graph_observed))
    with observe('fallback'):
        fallback_observed = fallback()
    check('observation_preserves_fallback_output', compare('observed_fallback_control', fallback_ref, fallback_observed))
    report['observed_metadata'] = observed_metadata
    modules('outer_observation')
    save()

    first = None
    for index in range(len(model.transformer_blocks)):
        incoming = compare(f'b{index}_input', buffers['graph'][f'b{index}_input'], buffers['fallback'][f'b{index}_input'])
        outgoing = compare(f'b{index}_output', buffers['graph'][f'b{index}_output'], buffers['fallback'][f'b{index}_output'])
        if not incoming or not outgoing:
            first = index
            report['first_differing_block'] = dict(index=index, input_equal=incoming, output_equal=outgoing)
            save()
            break
    if first is None:
        report['next_boundary'] = 'All compiled block boundaries match; inspect final norm/projection next.'
    else:
        # Modulation/frequencies/mask are shared across blocks. Reuse saved
        # copies from block zero while retaining each block's own immutable KV.
        replay = {}
        for arm in ('graph', 'fallback'):
            kw = dict(arguments[arm][first])
            kw['hidden_states'] = buffers[arm][f'b{first}_input']
            for field in ('modulation', 'freqs', 'target_token_mask'):
                kw[field] = buffers[arm][field]
            if kw.get('prepared_modulation') is not None:
                kw['prepared_modulation'] = tuple(buffers[arm][f'prepared_{n}'] for n in range(4))
            replay[arm] = kw
            for field in ('hidden_states', 'modulation', 'freqs', 'target_token_mask'):
                artifact(arm + '_' + field, kw[field])
            for kind, tensor in kw['kv_cache']['uncond'].items():
                artifact(arm + '_prefix_' + kind, tensor)
            artifact(arm + '_block_output', buffers[arm][f'b{first}_output'])
        for n, tensor in enumerate(replay['fallback'].get('prepared_modulation', ())):
            artifact('fallback_prepared_' + str(n), tensor)
        common = []
        for field in ('hidden_states', 'modulation', 'freqs', 'target_token_mask'):
            common.append(compare('common_' + field, replay['graph'][field], replay['fallback'][field]))
        for field in ('key', 'value'):
            common.append(compare('common_prefix_' + field, replay['graph']['kv_cache']['uncond'][field],
                                  replay['fallback']['kv_cache']['uncond'][field]))
        report['block_flags'] = {arm: {field: str(kw.get(field, False if field == '_pointwise_validated' else None))
                                       for field in ('attn_metadata', 'sp_prefix_len', 'sp_decode',
                                                     'cache_branch', 'cache_write_len', '_pointwise_validated')}
                                 for arm, kw in replay.items()}
        save()
        if all(common):
            for arm in ('graph', 'fallback'):
                with torch.inference_mode(report['inference_contexts'][arm]), torch.no_grad():
                    output = originals[first](**replay[arm])
                check(arm + '_isolated_block_reproduces', compare(arm + '_block_replay',
                      output, buffers[arm][f'b{first}_output']))
                del output
            # Change only prepared_modulation; retain each arm's other operands,
            # inference context and validation flag. Then test the flag alone.
            cases = [
                ('graph_with_prepared', 'graph', {'prepared_modulation': replay['fallback']['prepared_modulation']}),
                ('fallback_without_prepared', 'fallback', {'prepared_modulation': None}),
                ('graph_validation_flag_only', 'graph', {'_pointwise_validated': replay['fallback'].get('_pointwise_validated', False)}),
                ('fallback_validation_flag_only', 'fallback', {'_pointwise_validated': replay['graph'].get('_pointwise_validated', False)}),
            ]
            for label, arm, change in cases:
                with torch.inference_mode(report['inference_contexts'][arm]), torch.no_grad():
                    output = originals[first](**dict(replay[arm], **change))
                artifact(label, output)
                compare(label + '_vs_graph', output, buffers['graph'][f'b{first}_output'])
                compare(label + '_vs_fallback', output, buffers['fallback'][f'b{first}_output'])
                del output
            modules('block_argument_swaps')
        else:
            report['next_boundary'] = 'Common block operands already differ; no modulation attribution or swaps.'

    after = {f'{i}/{kind}': digest(row['uncond'][kind])
             for i, row in enumerate(changed) for kind in ('key', 'value')}
    check('request_prefixes_unchanged', hashes == after)
    report['prefix_sha256_after'] = after
    native.clear()
    candidate.clear()
    changed.clear()
    report['status'] = 'completed'
    save()
