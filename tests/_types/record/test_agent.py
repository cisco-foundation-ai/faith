# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgentResult record type."""

import pytest

from faith._types.json_types import JsonValue
from faith._types.record.agent import AgentResult, StoppedReason
from tests._types.record.fake_agent_result_maker import (
    make_fake_agent_result as _result,
)

# The persisted form of a whole clean result: stopped_reason as its lowercase
# value, and no "error" key at all rather than an explicit null.
_SERIALIZED_CLEAN_RESULT = {
    "task_id": "task-1",
    "answer": "flag{x}",
    "messages": [{"role": "user", "content": "go"}],
    "tool_calls": [
        {"tool_name": "submit", "timestamp": 1.0, "args": {"answer": "flag{x}"}}
    ],
    "tool_responses": [
        {
            "tool_name": "submit",
            "timestamp": 1.1,
            "response_truncated": "ok",
            "response_length": 2,
        }
    ],
    "num_tool_calls": 1,
    "execution_time_seconds": 1.5,
    "stopped_reason": "max_calls",
    "final_turn_offered": False,
}


@pytest.mark.parametrize(
    "answer",
    ["flag{x}", ["a.py", "b.py"], {"k": 1}, 42, 3.14, True, None],
)
def test_agent_result_round_trips_answer_variants(answer: JsonValue) -> None:
    # answer is JsonValue: every JSON type must survive to_json/from_json intact.
    result = _result(answer, StoppedReason.SUBMITTED)
    assert AgentResult.from_json(result.to_json()) == result


def test_a_clean_result_serializes_to_its_log_form_and_back() -> None:
    """The whole persisted shape, so a serialization drift cannot hide."""
    result = _result("flag{x}", StoppedReason.MAX_CALLS)
    assert result.to_dict() == _SERIALIZED_CLEAN_RESULT
    assert AgentResult.from_dict(_SERIALIZED_CLEAN_RESULT) == result


@pytest.mark.parametrize(
    "stopped_reason,error",
    [
        (StoppedReason.SUBMITTED, "boom"),
        (StoppedReason.ERROR, None),
    ],
)
def test_an_error_inconsistent_with_the_reason_is_rejected(
    stopped_reason: StoppedReason, error: str | None
) -> None:
    with pytest.raises(ValueError) as excinfo:
        _result(None, stopped_reason, error=error)

    assert str(excinfo.value) == (
        f"error must be set if and only if stopped_reason is ERROR; "
        f"got stopped_reason={stopped_reason}, error={error!r}."
    )


def test_a_log_record_violating_the_invariant_fails_the_read() -> None:
    """__post_init__ also runs on from_dict, so a bad log never reaches grading."""
    with pytest.raises(ValueError) as excinfo:
        AgentResult.from_dict(_SERIALIZED_CLEAN_RESULT | {"stopped_reason": "error"})

    assert str(excinfo.value) == (
        "error must be set if and only if stopped_reason is ERROR; "
        "got stopped_reason=error, error=None."
    )
