# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-alignment tests for MiniCPM-o 4.5's native Talker."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _DUPLEX_CODEC_TOKENS_PER_CHUNK,
    _KV_CACHE_EPOCH,
    _KV_NEXT_POSITION,
    _KV_PREFILL_STARTED,
    _MAX_AUDIO_TOKENS_PER_CONDITION,
    MiniCPMO45OmniTTSForConditionalGeneration,
    _resolve_codec_sampling_params,
    _restore_weight_norm_weight,
)
from vllm_omni.utils.mm_outputs import to_payload_element

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeNativeTalker(nn.Module):
    has_preprocess = True

    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs = None

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return torch.ones(2, 4)


def test_wrapper_always_delegates_talker_to_native_ar_path() -> None:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "tts"
    model.talker = _FakeNativeTalker()

    output = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.arange(2),
        model_intermediate_buffer=[{"request_id": "req"}],
    )

    assert output.shape == (2, 4)
    assert model.talker.forward_kwargs["model_intermediate_buffer"][0]["request_id"] == "req"


def _make_talker() -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker._num_audio_tokens = 8
    talker._batch_stop_logits = None
    talker._request_generators = {}
    talker._request_audio_states = {}
    talker._deferred_cleanup_ids = set()
    talker._codec_seed = 42
    return talker


def _routed(output, index: int):
    return to_payload_element(
        output.multimodal_outputs,
        index,
        index,
        index + 1,
        seq_len=2,
        scheduled_seq_len=2,
    )


def test_audio_token_limit_matches_official_per_condition_limit() -> None:
    assert _MAX_AUDIO_TOKENS_PER_CONDITION == 500


def test_codec_sampling_defaults_ignore_nested_hf_sampling_fields() -> None:
    config = SimpleNamespace(
        tts_config=SimpleNamespace(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.0,
            seed=7,
        )
    )

    source, params = _resolve_codec_sampling_params(config)

    assert source == "vllm_omni_default"
    assert params == {
        "temperature": 0.8,
        "top_p": 0.85,
        "top_k": 25,
        "repetition_penalty": 1.05,
        "seed": 42,
    }


def test_codec_sampling_deploy_values_override_vllm_omni_defaults() -> None:
    config = SimpleNamespace(
        minicpmo_codec_sampling_params={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.1,
            "seed": 9,
        }
    )

    source, params = _resolve_codec_sampling_params(config)

    assert source == "vllm_omni_deploy"
    assert params == {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.1,
        "seed": 9,
    }


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"unknown": 1}, "Unknown MiniCPM codec sampling parameter"),
        ({1: 1}, "Unknown MiniCPM codec sampling parameter"),
        ({"temperature": 0}, "temperature must be finite and positive"),
        ({"top_p": 0}, "top_p must be in"),
        ({"top_k": 0}, "top_k must be a positive integer"),
        ({"repetition_penalty": float("nan")}, "repetition_penalty must be finite and positive"),
        ({"seed": 1.5}, "seed must be an integer"),
    ],
)
def test_codec_sampling_deploy_values_reject_invalid_overrides(override, error) -> None:
    config = SimpleNamespace(minicpmo_codec_sampling_params=override)

    with pytest.raises((TypeError, ValueError), match=error):
        _resolve_codec_sampling_params(config)


def test_codec_sampling_deploy_values_must_be_a_mapping() -> None:
    config = SimpleNamespace(minicpmo_codec_sampling_params=[("top_p", 0.95)])

    with pytest.raises(TypeError, match="must be a mapping"):
        _resolve_codec_sampling_params(config)


def test_sliding_recompute_matches_official_streaming_cadence() -> None:
    talker = _make_talker()
    talker._sliding_recompute_enabled = True
    talker._sliding_window_size = 4
    talker._sliding_recomputed_chunks = 1

    assert not talker._should_recompute_condition(0)
    assert talker._should_recompute_condition(1)
    assert talker._should_recompute_condition(2)
    assert talker._should_recompute_condition(3)
    assert talker._should_recompute_condition(4)


def test_sliding_recompute_cadence_is_disabled_without_opt_in() -> None:
    talker = _make_talker()
    talker._sliding_recompute_enabled = False
    talker._sliding_window_size = 2
    talker._sliding_recomputed_chunks = 1

    assert all(not talker._should_recompute_condition(index) for index in range(6))


def test_sliding_recompute_prefill_uses_full_previous_audio_context() -> None:
    talker = _make_talker()
    talker._sliding_recompute_enabled = True
    talker._sliding_window_size = 2
    talker._sliding_recomputed_chunks = 1
    talker.emb_text = nn.Embedding(1, 2)
    talker.emb_code = nn.ModuleList([nn.Embedding(64, 2)])
    previous_condition = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    current_condition = torch.tensor([[4.0, 4.0], [5.0, 5.0]])
    previous_codes = list(range(20))
    prompt_len = previous_condition.shape[0] + len(previous_codes) + current_condition.shape[0]
    state = {
        "mode": "streaming",
        "condition_chunks": [previous_condition, current_condition],
        "condition_chunk_index": 1,
        "condition_cursor": 0,
        "condition_step": 0,
        "conditioning": False,
        "finished": False,
        "sliding_recompute_pending": True,
        "sliding_recompute_prompt_len": prompt_len,
        "sliding_recompute_audio_tokens": len(previous_codes),
        "completed_condition_audio": [
            {"condition_index": 0, "codes": previous_codes},
        ],
    }
    talker._request_audio_states["req-recompute"] = state

    _, embeds, update = talker.preprocess(
        torch.zeros(prompt_len, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        _omni_prompt_len=prompt_len,
        request_id="req-recompute",
        audio_state=state,
    )

    expected = torch.cat(
        [previous_condition, talker.emb_code[0](torch.tensor(previous_codes)), current_condition],
        dim=0,
    )
    assert torch.equal(embeds, expected)
    assert update["audio_state"]["sliding_recompute_pending"] is False
    assert update["audio_state"]["condition_step"] == 0
    assert update["audio_state"]["prefill_source"] == "sliding_recompute"


def _make_sliding_prefill_state(*, prompt_len: int = 4) -> dict:
    return {
        "mode": "streaming",
        "condition_chunks": [torch.ones(1, 2), torch.zeros(1, 2)],
        "condition_chunk_index": 1,
        "condition_cursor": 0,
        "condition_step": 0,
        "conditioning": False,
        "finished": False,
        "sliding_recompute_pending": True,
        "sliding_recompute_prompt_len": prompt_len,
        "sliding_recompute_audio_tokens": prompt_len - 2,
        "completed_condition_audio": [{"condition_index": 0, "codes": list(range(prompt_len - 2))}],
        "recompute_epoch": 3,
        _KV_CACHE_EPOCH: 3,
        _KV_NEXT_POSITION: 0,
        _KV_PREFILL_STARTED: False,
    }


def test_sliding_recompute_rejects_nonzero_first_kv_position() -> None:
    talker = _make_talker()
    talker._sliding_recompute_enabled = True
    talker._sliding_window_size = 2
    talker._sliding_recomputed_chunks = 1
    talker.emb_text = nn.Embedding(1, 2)
    talker.emb_code = nn.ModuleList([nn.Embedding(64, 2)])
    state = _make_sliding_prefill_state()
    talker._request_audio_states["req-nonzero-start"] = state

    with pytest.raises(RuntimeError, match="position zero"):
        talker.preprocess(
            torch.zeros(1, dtype=torch.long),
            None,
            _omni_is_prefill=True,
            _omni_num_computed_tokens=1,
            _omni_prompt_len=4,
            request_id="req-nonzero-start",
            audio_state=state,
        )


def test_sliding_recompute_rejects_a_gap_between_prefill_spans() -> None:
    talker = _make_talker()
    talker._sliding_recompute_enabled = True
    talker._sliding_window_size = 2
    talker._sliding_recomputed_chunks = 1
    talker.emb_text = nn.Embedding(1, 2)
    talker.emb_code = nn.ModuleList([nn.Embedding(64, 2)])
    state = _make_sliding_prefill_state()
    talker._request_audio_states["req-prefill-gap"] = state

    talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        _omni_prompt_len=4,
        request_id="req-prefill-gap",
        audio_state=state,
    )

    with pytest.raises(RuntimeError, match="not contiguous"):
        talker.preprocess(
            torch.zeros(1, dtype=torch.long),
            None,
            _omni_is_prefill=True,
            _omni_num_computed_tokens=3,
            _omni_prompt_len=4,
            request_id="req-prefill-gap",
            audio_state=state,
        )


def test_sliding_recompute_rejects_epoch_divergence() -> None:
    talker = _make_talker()
    talker._sliding_recompute_enabled = True
    talker._sliding_window_size = 2
    talker._sliding_recomputed_chunks = 1
    talker.emb_text = nn.Embedding(1, 2)
    talker.emb_code = nn.ModuleList([nn.Embedding(64, 2)])
    state = _make_sliding_prefill_state()
    state[_KV_CACHE_EPOCH] = 2
    talker._request_audio_states["req-epoch-drift"] = state

    with pytest.raises(RuntimeError, match="epoch"):
        talker.preprocess(
            torch.zeros(1, dtype=torch.long),
            None,
            _omni_is_prefill=True,
            _omni_num_computed_tokens=0,
            _omni_prompt_len=4,
            request_id="req-epoch-drift",
            audio_state=state,
        )


def test_sliding_recompute_rejects_runner_position_mismatch() -> None:
    talker = _make_talker()
    talker._sliding_recompute_enabled = True
    talker._sliding_window_size = 2
    talker._sliding_recomputed_chunks = 1
    talker.emb_text = nn.Embedding(1, 2)
    talker.emb_code = nn.ModuleList([nn.Embedding(64, 2)])
    state = _make_sliding_prefill_state()
    talker._request_audio_states["req-position-mismatch"] = state

    with pytest.raises(RuntimeError, match="runner position"):
        talker.preprocess(
            torch.zeros(1, dtype=torch.long),
            None,
            _omni_is_prefill=True,
            _omni_num_computed_tokens=0,
            _omni_position_start=1,
            _omni_position_end=1,
            _omni_prompt_len=4,
            request_id="req-position-mismatch",
            audio_state=state,
        )


def test_native_decode_ignores_stale_prefill_runner_positions() -> None:
    talker = _make_talker()
    state = {
        "recompute_epoch": 0,
        _KV_CACHE_EPOCH: 0,
        _KV_NEXT_POSITION: 0,
        _KV_PREFILL_STARTED: False,
        "initial_prefill_pending": True,
    }
    prefill_info = {
        "_omni_num_computed_tokens": 0,
        "_omni_position_start": 0,
        "_omni_position_end": 1,
    }

    talker._validate_kv_input_progress(
        state,
        prefill_info,
        span_len=2,
        is_prefill=True,
        request_id="req-stale-runner-position",
        source="initial_condition",
    )

    stale_decode_info = {
        **prefill_info,
        "_omni_num_computed_tokens": 2,
    }
    talker._validate_kv_input_progress(
        state,
        stale_decode_info,
        span_len=1,
        is_prefill=False,
        request_id="req-stale-runner-position",
        source="native_kv_decode",
    )

    assert state[_KV_NEXT_POSITION] == 3


@pytest.mark.parametrize(
    ("expected_prompt_len", "runner_prompt_len", "is_prefill", "error"),
    [
        (6, 5, True, "runner prompt"),
        (7, 7, True, "does not match its rebuilt condition"),
        (6, 6, False, "prefill boundary"),
    ],
)
def test_sliding_recompute_rejects_invalid_session_boundary(
    expected_prompt_len: int,
    runner_prompt_len: int,
    is_prefill: bool,
    error: str,
) -> None:
    talker = _make_talker()
    talker._sliding_recompute_enabled = True
    talker._sliding_window_size = 2
    talker._sliding_recomputed_chunks = 1
    talker.emb_text = nn.Embedding(1, 2)
    talker.emb_code = nn.ModuleList([nn.Embedding(64, 2)])
    state = {
        "mode": "streaming",
        "condition_chunks": [torch.ones(2, 2), torch.ones(3, 2)],
        "condition_chunk_index": 1,
        "condition_cursor": 0,
        "condition_step": 0,
        "conditioning": False,
        "finished": False,
        "sliding_recompute_pending": True,
        "sliding_recompute_prompt_len": expected_prompt_len,
        "sliding_recompute_audio_tokens": 1,
        "completed_condition_audio": [{"condition_index": 0, "codes": [1]}],
    }
    talker._request_audio_states["req-invalid-boundary"] = state

    with pytest.raises(ValueError, match=error):
        talker.preprocess(
            torch.zeros(1, dtype=torch.long),
            None,
            _omni_is_prefill=is_prefill,
            _omni_num_computed_tokens=0,
            _omni_prompt_len=runner_prompt_len,
            request_id="req-invalid-boundary",
            audio_state=state,
        )


def test_sliding_recompute_normalizes_transport_condition_device() -> None:
    talker = _make_talker()
    talker._sliding_recompute_enabled = True
    talker._sliding_window_size = 2
    talker._sliding_recomputed_chunks = 1
    talker.emb_text = nn.Embedding(1, 2).to("meta")
    talker.emb_code = nn.ModuleList([nn.Embedding(64, 2).to("meta")])
    state = {
        "mode": "streaming",
        "condition_chunk_index": 1,
        "condition_cursor": 0,
        "condition_step": 0,
        "conditioning": False,
        "finished": False,
        "sliding_recompute_pending": True,
        "sliding_recompute_prompt_len": 4,
        "sliding_recompute_audio_tokens": 2,
        "completed_condition_audio": [{"condition_index": 0, "codes": [1, 2]}],
        "condition_chunks": [torch.ones(1, 2), torch.zeros(1, 2)],
    }
    talker._request_audio_states["req-device"] = state

    _, embeds, _ = talker.preprocess(
        torch.zeros(4, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        _omni_prompt_len=4,
        request_id="req-device",
        audio_state=state,
    )

    assert embeds.device.type == "meta"
    assert all(chunk.device.type == "meta" for chunk in state["condition_chunks"])


def test_sliding_recompute_emits_prompt_replacement_at_condition_boundary(mocker) -> None:
    talker = _make_talker()
    talker._sliding_recompute_enabled = True
    talker._sliding_window_size = 2
    talker._sliding_recomputed_chunks = 1
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(3))
    chunks = [torch.zeros(2, 2), torch.ones(3, 2), torch.full((4, 2), 2.0)]
    state = {
        "mode": "streaming",
        "step": 10,
        "condition_step": _MAX_AUDIO_TOKENS_PER_CONDITION - 1,
        "finished": False,
        "condition_chunks": chunks,
        "condition_chunk_index": 1,
        "condition_cursor": 0,
        "conditioning": False,
        "condition_audio_codes": list(range(_MAX_AUDIO_TOKENS_PER_CONDITION - 1)),
        "sliding_recompute_audio_tokens": 2,
        "completed_condition_audio": [{"condition_index": 0, "codes": [1, 2]}],
    }
    talker._request_audio_states["req-recompute-boundary"] = state
    info = {
        "request_id": "req-recompute-boundary",
        "audio_state": state,
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    meta = output.multimodal_outputs["meta"]
    assert meta["replace_streaming_prompt"][0].item() is True
    assert meta["kv_cache_epoch"][0].item() == 1
    assert meta["next_stage_prompt_len"][0].item() == (
        chunks[1].shape[0] + _MAX_AUDIO_TOKENS_PER_CONDITION + chunks[2].shape[0]
    )
    routed = _routed(output, 0)
    assert routed["meta"]["replace_streaming_prompt"].item() is True
    assert routed["meta"]["next_stage_prompt_len"].item() == meta["next_stage_prompt_len"][0].item()
    assert state["condition_chunk_index"] == 2
    assert state["condition_step"] == 0
    assert state["sliding_recompute_pending"] is True
    assert state[_KV_CACHE_EPOCH] == 1
    assert state[_KV_NEXT_POSITION] == 0
    assert state[_KV_PREFILL_STARTED] is False
    assert state["completed_condition_audio"][-1] == {
        "condition_index": 1,
        "codes": [*range(_MAX_AUDIO_TOKENS_PER_CONDITION - 1), 3],
    }


def test_condition_chunks_match_official_streaming_boundaries() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(32, 4)
    talker.projector_semantic = nn.Identity()
    talker._normalize = False
    talker._text_eos_id = 30
    talker._tts_bos_id = 31
    token_ids = torch.arange(23)
    hidden_states = torch.zeros(23, 4)

    chunks = talker._build_condition_chunks(token_ids, hidden_states)

    assert [chunk.shape[0] for chunk in chunks] == [11, 11, 5]
    assert torch.equal(chunks[0][-1], talker.emb_text.weight[31])
    assert torch.equal(chunks[1][-1], talker.emb_text.weight[31])
    assert torch.equal(chunks[2][-2], talker.emb_text.weight[30])
    assert torch.equal(chunks[2][-1], talker.emb_text.weight[31])


def test_first_audio_token_skips_sampling_processors(mocker) -> None:
    talker = _make_talker()
    talker.head_code = nn.ModuleList([nn.Identity()])
    talker._codec_temperature = 0.8
    talker._codec_top_k = 3
    talker._codec_top_p = 0.85
    talker._codec_repetition_penalty = 1.05
    talker._codec_seed = 42
    repetition_penalty = mocker.spy(minicpmo_4_5_omni_tts, "_apply_repetition_penalty")
    top_k_top_p = mocker.spy(minicpmo_4_5_omni_tts, "_apply_top_k_top_p")
    hidden = torch.arange(8, dtype=torch.float32).unsqueeze(0)

    talker._sample_audio_code(
        hidden,
        torch.empty(0, dtype=torch.long),
        "req-reference-parity",
        step=0,
    )

    repetition_penalty.assert_not_called()
    top_k_top_p.assert_not_called()

    talker._sample_audio_code(
        hidden,
        torch.tensor([1]),
        "req-reference-parity",
        step=1,
    )

    repetition_penalty.assert_called_once()
    top_k_top_p.assert_called_once()


def test_codec_sampling_diagnostics_report_filter_and_recompute_state(mocker) -> None:
    talker = _make_talker()
    talker.head_code = nn.ModuleList([nn.Identity()])
    talker._codec_temperature = 0.8
    talker._codec_top_k = 3
    talker._codec_top_p = 0.85
    talker._codec_repetition_penalty = 1.05
    talker._codec_seed = 42
    talker._request_audio_states["req-diagnostics"] = {
        "mode": "streaming",
        "condition_chunk_index": 14,
        "condition_step": 24,
        "prefill_source": "sliding_recompute",
        "recompute_epoch": 2,
    }
    info = mocker.patch.object(minicpmo_4_5_omni_tts.logger, "info")

    talker._sample_audio_code(
        torch.tensor([[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 0.0]]),
        torch.tensor([1]),
        "req-diagnostics",
        step=99,
    )

    diagnostic_calls = [call for call in info.call_args_list if "[codec-sampling]" in call.args[0]]
    assert len(diagnostic_calls) == 1
    args = diagnostic_calls[0].args
    assert args[1:4] == ("req-diagnostics", "streaming", "sliding_recompute")
    assert args[5:7] == (14, 25)
    assert args[19] is False
    assert args[20] is True
    assert args[21] is True


def test_codec_sampling_diagnostics_are_bounded_to_selected_steps(mocker) -> None:
    talker = _make_talker()
    talker.head_code = nn.ModuleList([nn.Identity()])
    talker._codec_temperature = 0.8
    talker._codec_top_k = 3
    talker._codec_top_p = 0.85
    talker._codec_repetition_penalty = 1.05
    talker._codec_seed = 42
    talker._request_audio_states["req-diagnostics-bounded"] = {
        "mode": "streaming",
        "condition_chunk_index": 14,
        "condition_step": 25,
        "prefill_source": "sliding_recompute",
    }
    info = mocker.patch.object(minicpmo_4_5_omni_tts.logger, "info")

    talker._sample_audio_code(
        torch.ones(1, 8),
        torch.tensor([1]),
        "req-diagnostics-bounded",
        step=100,
    )

    assert not any("[codec-sampling]" in call.args[0] for call in info.call_args_list)


def test_audio_eos_is_not_masked_on_first_sampling_step(mocker) -> None:
    talker = _make_talker()
    talker.head_code = nn.ModuleList([nn.Identity()])
    talker._codec_temperature = 0.8
    talker._codec_top_k = 3
    talker._codec_top_p = 0.85
    talker._codec_repetition_penalty = 1.05
    talker._codec_seed = 42

    def sample(probabilities, **_):
        assert probabilities[0, -1] > 0
        return torch.tensor([[7]])

    mocker.patch("torch.multinomial", side_effect=sample)

    sampled = talker._sample_audio_code(
        torch.arange(8, dtype=torch.float32).unsqueeze(0),
        torch.empty(0, dtype=torch.long),
        "req-eos-eligible",
        step=0,
    )

    assert sampled.item() == 7


def test_weight_norm_restore_matches_checkpoint_parametrization_in_bfloat16() -> None:
    generator = torch.Generator().manual_seed(42)
    weight_v = torch.randn(8, 16, generator=generator, dtype=torch.bfloat16)
    weight_g = torch.rand(8, 1, generator=generator, dtype=torch.bfloat16)
    linear = nn.utils.parametrizations.weight_norm(
        nn.Linear(16, 8, bias=False, dtype=torch.bfloat16),
        dim=0,
    )
    with torch.no_grad():
        linear.parametrizations.weight.original0.copy_(weight_g)
        linear.parametrizations.weight.original1.copy_(weight_v)

    restored = _restore_weight_norm_weight(weight_g, weight_v)

    assert torch.equal(restored, linear.weight)


def test_talker_emits_request_aligned_codec_deltas_after_compaction(mocker) -> None:
    talker = _make_talker()
    seen: list[tuple[str, list[float], list[int]]] = []

    def sample(hidden, history, request_id, step):
        assert step == 0
        seen.append((request_id, hidden.reshape(-1).tolist(), history.tolist()))
        return torch.tensor(2 if request_id == "req-a" else 3)

    mocker.patch.object(talker, "_sample_audio_code", side_effect=sample)
    infos = [
        {"request_id": "req-a", "audio_codes": {"accumulated": torch.tensor([1])}},
        {"request_id": "req-b", "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)}},
    ]

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
    )

    assert seen == [
        ("req-a", [2.0, 0.0], [1]),
        ("req-b", [3.0, 0.0], []),
    ]
    assert infos[0]["audio_codes"]["accumulated"].tolist() == [1, 2]
    assert infos[1]["audio_codes"]["accumulated"].tolist() == [3]
    assert set(output.multimodal_outputs) == {"codes", "meta"}
    assert "model_outputs" not in output.multimodal_outputs
    assert "sr" not in output.multimodal_outputs
    assert _routed(output, 0)["codes"]["audio"].tolist() == [[2]]
    assert _routed(output, 1)["codes"]["audio"].tolist() == [[3]]
    assert _routed(output, 0)["meta"]["finished"].item() is False
    assert set(output.multimodal_outputs["meta"]) == {"finished"}
    assert talker.compute_logits(output.text_hidden_states).argmax(dim=-1).tolist() == [0, 0]


def test_talker_projects_request_aligned_duplex_metadata(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(2))
    infos = [
        {
            "request_id": "req-a",
            "native_duplex": True,
            "duplex": {"epoch": 3, "turn_id": 7},
            "ids": {"tts": [41]},
            "meta": {
                "native_duplex_segment_text": "first",
                "turn_eos_token_id": 99,
            },
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
        {
            "request_id": "req-b",
            "native_duplex": True,
            "duplex": {"epoch": 4, "turn_id": 8},
            "ids": {"tts": [42, 99]},
            "meta": {
                "native_duplex_segment_text": "second",
                "turn_eos_token_id": 99,
            },
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
    ]

    output = talker.make_omni_output(
        torch.ones(2, 2),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 1), (1, 2)],
    )

    meta = output.multimodal_outputs["meta"]
    assert [value.item() for value in meta["native_duplex"]] == [True, True]
    assert [value.item() for value in meta["duplex_epoch"]] == [3, 4]
    assert [value.item() for value in meta["duplex_turn_id"]] == [7, 8]
    assert "native_duplex_segment_text" not in meta
    assert [bytes(value.tolist()).decode("utf-8") for value in meta["llm_output_text_utf8"]] == [
        "first",
        "second",
    ]
    assert [value.item() for value in meta["turn_end"]] == [False, True]


def test_talker_rejects_native_duplex_without_fence_identity(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(2))
    info = {
        "request_id": "req-missing-fence",
        "native_duplex": True,
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    with pytest.raises(RuntimeError, match="requires non-negative integer epoch and turn_id"):
        talker.make_omni_output(
            torch.ones(1, 2),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, 1)],
        )


def test_incomplete_prefill_emits_no_code_and_does_not_advance_state(mocker) -> None:
    talker = _make_talker()
    sample = mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(2))
    infos = [
        {
            "request_id": "req-prefill",
            "audio_state": {"step": 0},
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
        {
            "request_id": "req-decode",
            "audio_state": {"step": 4},
            "audio_codes": {"accumulated": torch.tensor([1])},
        },
    ]

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
        request_sample_eligible=[False, True],
    )

    sample.assert_called_once()
    assert sample.call_args.args[2] == "req-decode"
    assert infos[0]["audio_state"]["step"] == 0
    assert infos[0]["audio_codes"]["accumulated"].numel() == 0
    assert infos[1]["audio_state"]["step"] == 5
    assert _routed(output, 0)["codes"]["audio"].shape == (0, 1)
    assert _routed(output, 1)["codes"]["audio"].tolist() == [[2]]


def test_eos_is_terminal_once_and_never_enters_codec_history(mocker) -> None:
    talker = _make_talker()
    sample = mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(7))
    info = {
        "request_id": "req-stop",
        "audio_state": {"step": 3},
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }

    first = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    first_logits = talker.compute_logits(first.text_hidden_states)
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    sample.assert_called_once()
    assert info["audio_codes"]["accumulated"].tolist() == [4, 5]
    assert first.multimodal_outputs["codes"]["audio"][0].shape == (0, 1)
    assert first.multimodal_outputs["meta"]["finished"][0].item() is True
    assert second.multimodal_outputs["meta"]["finished"][0].item() is False
    assert first_logits.argmax(dim=-1).tolist() == [1]
    assert talker.compute_logits(second.text_hidden_states).argmax(dim=-1).tolist() == [1]


def test_intermediate_eos_prefills_next_condition_before_sampling(mocker) -> None:
    talker = _make_talker()
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 2)])
    sample = mocker.patch.object(
        talker,
        "_sample_audio_code",
        side_effect=[torch.tensor(7), torch.tensor(3)],
    )
    next_condition = torch.tensor([[10.0, 0.0], [20.0, 0.0]])
    state = {
        "mode": "streaming",
        "step": 3,
        "condition_step": 0,
        "finished": False,
        "condition_chunks": [torch.zeros(1, 2), next_condition],
        "condition_chunk_index": 0,
        "condition_cursor": 0,
        "conditioning": False,
    }
    info = {
        "request_id": "req-chunked",
        "audio_state": state,
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }
    talker._request_audio_states["req-chunked"] = state

    eos_output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    assert eos_output.multimodal_outputs["meta"]["finished"][0].item() is False
    assert talker.compute_logits(eos_output.text_hidden_states).argmax(dim=-1).tolist() == [0]
    assert state["conditioning"] is True
    assert state["condition_chunk_index"] == 1
    assert state["condition_step"] == 0

    _, first_embed, _ = talker.preprocess(
        torch.tensor([0]),
        None,
        request_id="req-chunked",
        audio_state=state,
        audio_codes=info["audio_codes"],
    )
    first_condition_output = talker.make_omni_output(
        first_embed,
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    assert torch.equal(first_embed, next_condition[:1])
    assert first_condition_output.multimodal_outputs["codes"]["audio"][0].numel() == 0
    assert sample.call_count == 1

    _, final_embed, _ = talker.preprocess(
        torch.tensor([0]),
        None,
        request_id="req-chunked",
        audio_state=state,
        audio_codes=info["audio_codes"],
    )
    first_code_output = talker.make_omni_output(
        final_embed,
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    assert torch.equal(final_embed, next_condition[1:])
    assert first_code_output.multimodal_outputs["codes"]["audio"][0].tolist() == [[3]]
    assert state["conditioning"] is False
    assert state["step"] == 5
    assert sample.call_count == 2
    assert sample.call_args_list[1].args[3] == 4


def test_condition_limit_prefills_next_condition_without_finishing_request(mocker) -> None:
    talker = _make_talker()
    sample = mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(3))
    state = {
        "mode": "streaming",
        "step": 700,
        "condition_step": _MAX_AUDIO_TOKENS_PER_CONDITION - 1,
        "finished": False,
        "condition_chunks": [torch.zeros(1, 2), torch.ones(1, 2)],
        "condition_chunk_index": 0,
        "condition_cursor": 0,
        "conditioning": False,
    }
    info = {
        "request_id": "req-condition-limit",
        "audio_state": state,
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    assert output.multimodal_outputs["codes"]["audio"][0].tolist() == [[3]]
    assert output.multimodal_outputs["meta"]["finished"][0].item() is False
    assert talker.compute_logits(output.text_hidden_states).argmax(dim=-1).tolist() == [0]
    assert info["audio_codes"]["accumulated"].tolist() == [4, 5, 3]
    request_state = info["audio_state"]
    assert request_state["step"] == 701
    assert request_state["condition_step"] == 0
    assert request_state["condition_chunk_index"] == 1
    assert request_state["conditioning"] is True
    assert sample.call_args.args[3] == 700


def test_final_condition_limit_is_terminal_and_includes_new_codec_delta(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(3))
    info = {
        "request_id": "req-limit",
        "audio_state": {
            "mode": "streaming",
            "step": 1,
            "condition_step": _MAX_AUDIO_TOKENS_PER_CONDITION - 1,
        },
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    assert info["audio_codes"]["accumulated"].tolist() == [4, 5, 3]
    assert output.multimodal_outputs["codes"]["audio"][0].tolist() == [[3]]
    assert output.multimodal_outputs["meta"]["finished"][0].item() is True
    assert talker.compute_logits(output.text_hidden_states).argmax(dim=-1).tolist() == [1]


def test_request_local_state_survives_missing_runner_buffer_update(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(
        talker,
        "_sample_audio_code",
        side_effect=[torch.tensor(3), torch.tensor(7)],
    )
    first_info = {
        "request_id": "req-local-state",
        "audio_state": {"mode": "streaming", "step": 1, "condition_step": 1},
        "audio_codes": {"accumulated": torch.tensor([4])},
    }

    talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[first_info],
        request_token_spans=[(0, 1)],
    )
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[{"request_id": "req-local-state"}],
        request_token_spans=[(0, 1)],
    )

    assert second.multimodal_outputs["meta"]["finished"][0].item() is True
    assert talker._request_audio_states["req-local-state"]["step"] == 3


def test_missing_conditioning_fails_clearly() -> None:
    talker = _make_talker()

    with pytest.raises(ValueError, match="tts_token_ids and tts_hidden_states"):
        talker.preprocess(
            torch.tensor([0]),
            None,
            _omni_is_prefill=True,
            request_id="req-invalid",
        )


def test_empty_speech_segment_finishes_without_sampling_codes() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(8, 4)
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 4)])
    talker._text_eos_id = 5
    talker._tts_bos_id = 6

    _, embeds, updates = talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-empty",
        tts_token_ids=torch.empty(0, dtype=torch.long),
        tts_hidden_states=torch.empty(0, 4),
    )

    assert torch.equal(embeds, talker.emb_text(torch.tensor([5, 6])))
    assert updates["audio_state"]["finished"] is True

    # A stale decode after request completion still has no previous codec token
    # to embed, so the empty-condition fallback must remain shape-correct.
    _, decode_embeds, _ = talker.preprocess(
        torch.zeros(1, dtype=torch.long),
        None,
        request_id="req-empty",
        audio_state=updates["audio_state"],
        audio_codes=updates["audio_codes"],
    )

    assert decode_embeds.shape == (1, 4)


def test_chunked_prefill_tail_aligns_condition_with_prompt_length(mocker) -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    with torch.no_grad():
        talker.emb_text.weight.copy_(torch.tensor([[100.0, 101.0]]))
    condition = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    mocker.patch.object(talker, "_build_condition_chunks", return_value=[condition])

    _, embeds, _ = talker.preprocess(
        torch.zeros(12, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        _omni_prompt_len=12,
        request_id="req-chunked-prefill",
        tts_token_ids=torch.tensor([1]),
        tts_hidden_states=torch.ones(1, 2),
    )

    expected = torch.cat([talker.emb_text(torch.zeros(3, dtype=torch.long)), condition], dim=0)
    assert torch.equal(embeds, expected)
    state = talker._request_audio_states["req-chunked-prefill"]
    assert state["condition_step"] == 0
    assert state["condition_chunk_index"] == 0
    assert state["prefill_source"] == "initial_condition"


@pytest.mark.parametrize(
    ("meta", "expected_min_tokens"),
    [
        ({"turn_start": True}, 0),
        ({}, _DUPLEX_CODEC_TOKENS_PER_CHUNK),
        ({"turn_end": True}, 0),
    ],
)
def test_native_duplex_prefill_uses_official_chunk_limits(
    mocker,
    meta,
    expected_min_tokens,
) -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    mocker.patch.object(
        talker,
        "_build_condition_chunks",
        return_value=[torch.ones(3, 2)],
    )

    talker.preprocess(
        torch.zeros(3, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-duplex-chunk",
        native_duplex=True,
        meta=meta,
        tts_token_ids=torch.tensor([1]),
        tts_hidden_states=torch.ones(1, 2),
    )

    state = talker._request_audio_states["req-duplex-chunk"]
    assert state["mode"] == "native_duplex"
    assert state["min_tokens"] == expected_min_tokens
    assert state["max_tokens"] == _DUPLEX_CODEC_TOKENS_PER_CHUNK


def test_native_duplex_condition_matches_official_text_plus_audio_bos() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(16, 2)
    talker.projector_semantic = nn.Identity()
    talker._normalize = False
    talker._text_eos_id = 14
    talker._tts_bos_id = 15
    with torch.no_grad():
        talker.emb_text.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(16, 2))

    token_ids = torch.tensor([2, 3])
    hidden_states = torch.tensor([[0.5, 1.0], [1.5, 2.0]])

    chunks = talker._build_condition_chunks(
        token_ids,
        hidden_states,
        native_duplex=True,
    )
    condition = chunks[0]

    expected_text = talker.emb_text(token_ids) + hidden_states
    expected = torch.cat(
        [expected_text, talker._audio_bos_embedding()],
        dim=0,
    )
    assert torch.equal(condition, expected)
    assert condition.shape[0] == token_ids.shape[0] + 1


def test_request_cleanup_evicts_ar_rng_and_decode_state() -> None:
    talker = _make_talker()
    talker._request_generators["req-done"] = torch.Generator()
    talker._request_audio_states["req-done"] = {"step": 1}

    talker.on_requests_finished(["req-done"])
    talker._flush_deferred_cleanup()

    assert "req-done" not in talker._request_generators
    assert "req-done" not in talker._request_audio_states
