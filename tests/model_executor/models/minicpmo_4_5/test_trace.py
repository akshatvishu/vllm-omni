# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.trace import (
    int_sequence_summary,
    tensor_summary,
    text_summary,
    trace_event,
    update_int_hash,
    waveform_summary,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_trace_event_is_opt_in_and_json_parseable(monkeypatch, caplog) -> None:
    logger = logging.getLogger("minicpmo45-trace-test")
    caplog.set_level(logging.INFO, logger=logger.name)

    trace_event(logger, "disabled", request_id="req")
    assert not caplog.records

    monkeypatch.setenv("MINICPMO45_TRACE", "1")
    trace_event(logger, "enabled", request_id="req", count=3)
    trace_event(logger, "fallback", value=object())

    assert len(caplog.records) == 2
    prefix, payload = caplog.records[0].message.split(" ", 1)
    assert prefix == "[MiniCPMO45Trace]"
    assert json.loads(payload) == {
        "count": 3,
        "event": "enabled",
        "request_id": "req",
    }
    assert "object at " in json.loads(caplog.records[1].message.split(" ", 1)[1])["value"]


def test_trace_summaries_and_incremental_codec_hash_are_stable() -> None:
    token_summary = int_sequence_summary([1, 2, 3])
    assert token_summary["count"] == 3
    assert token_summary["head"] == [1, 2, 3]
    assert token_summary["tail"] == [1, 2, 3]
    assert token_summary == int_sequence_summary(torch.tensor([1, 2, 3]))

    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    assert tensor_summary(tensor) == tensor_summary(tensor.clone())

    bfloat16_tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert tensor_summary(bfloat16_tensor) == tensor_summary(bfloat16_tensor.clone())

    waveform = waveform_summary(torch.tensor([0.0, 1.0, -1.0, float("nan")]))
    assert waveform["samples"] == 4
    assert waveform["minimum"] == -1.0
    assert waveform["maximum"] == 1.0
    assert waveform["nonfinite"] == 1

    assert update_int_hash(update_int_hash(None, [1, 2]), [3, 4]) == update_int_hash(None, [1, 2, 3, 4])
    assert text_summary("hello")["tail"] == "hello"
