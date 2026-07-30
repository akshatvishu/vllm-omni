# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.helpers.assertions import (
    _assert_long_form_requirements,
    _assert_transcript_matches,
    _text_audio_mismatch_details,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_short_transcript_repeat_passes_containment_fallback():
    _assert_transcript_matches(
        " How... how are you?",
        audio_bytes=None,
        expected_text="how are you",
        threshold=0.9,
    )


def test_short_transcript_unrelated_text_still_fails():
    with pytest.raises(AssertionError, match="Transcript doesn't match input"):
        _assert_transcript_matches(
            " I don't know, sorry.",
            audio_bytes=None,
            expected_text="how are you",
            threshold=0.9,
        )


def test_text_audio_mismatch_details_include_lengths_and_tails():
    details = _text_audio_mismatch_details(
        text="begin text ending",
        transcript="begin audio ending",
        similarity=0.5,
        threshold=0.8,
        audio_bytes=b"wav",
    )

    assert "similarity=0.500000" in details
    assert "threshold=0.800000" in details
    assert "text_chars=17" in details
    assert "transcript_chars=18" in details
    assert "audio_bytes=3" in details
    assert "text_tail='begin text ending'" in details
    assert "transcript_tail='begin audio ending'" in details


def test_long_form_requirements_check_length_and_audio_tail():
    closing = "The lantern went dark and the long journey was complete"
    story = " ".join(["word"] * 500) + f" {closing}."

    _assert_long_form_requirements(
        text=story,
        transcript=f"The narrator finished. {closing}.",
        request_config={
            "minimum_text_words": 500,
            "required_text_suffix": closing,
            "required_audio_text": closing,
        },
    )

    with pytest.raises(AssertionError, match="Audio output is missing required text"):
        _assert_long_form_requirements(
            text=story,
            transcript="The audio stopped before the ending.",
            request_config={
                "minimum_text_words": 500,
                "required_text_suffix": closing,
                "required_audio_text": closing,
            },
        )
