# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Validation rejects calls before charging tools or contacting an endpoint."""

import json
from unittest.mock import Mock, patch

import httpx
import pytest

from faith._types.model.generation import GenParams
from faith._types.record.agent import StoppedReason
from faith.agentic.core import ReActLoop, ToolRegistry
from faith.agentic.tools.bash_exec import BashExecTool
from faith.agentic.tools.submit import SubmitTool
from faith.cli.subcmd.query import _check_tool_calling
from tests.agentic.core.test_react_loop import (
    _SUBMIT_TURN,
    _TASK,
    _FakeEnvironment,
    _FakeModelAPI,
    _loop,
    _turn,
)
from tests.model.test_api_retry import _completion, _model


@pytest.mark.parametrize("same_turn", [False, True])
def test_missing_arguments_leave_budget_for_a_corrected_call(same_turn: bool) -> None:
    bad = ("bad", "bash_exec", "{}")
    good = ("good", "bash_exec", '{"command": "ls"}')
    turns = [_turn("", bad, good)] if same_turn else [_turn("", bad), _turn("", good)]
    model = _FakeModelAPI([*turns, _SUBMIT_TURN])
    env = _FakeEnvironment()
    result = _loop(
        model,
        ToolRegistry([BashExecTool(), SubmitTool()]),
        env,
        max_tool_calls=1,
        max_turns=3,
    ).run(_TASK)
    assert env.commands_executed == ["ls"]
    assert result.stopped_reason == StoppedReason.SUBMITTED
    assert result.num_tool_calls == 3
    assert result.tool_calls[0]["error"] == "invalid_args"
    assert "error" not in result.tool_calls[1]
    assert result.tool_responses[0]["response_truncated"] == (
        "Error: bash_exec requires 'command' argument"
    )
    assert "[0 tool-calls remaining]" in result.tool_responses[1]["response_truncated"]


def test_repeated_validation_rejections_are_bounded_by_turns() -> None:
    model = _FakeModelAPI([_turn("", ("bad", "bash_exec", "{}"))] * 3)
    env = _FakeEnvironment()
    result = _loop(
        model,
        ToolRegistry([BashExecTool(), SubmitTool()]),
        env,
        max_tool_calls=1,
        max_turns=3,
    ).run(_TASK)
    assert result.stopped_reason == StoppedReason.MAX_TURNS
    assert not result.final_turn_offered
    assert not env.commands_executed
    assert result.num_tool_calls == 3
    assert all(call["error"] == "invalid_args" for call in result.tool_calls)


@pytest.mark.parametrize("raises", [False, True])
def test_executed_tool_errors_still_consume_the_budget(raises: bool) -> None:
    model = _FakeModelAPI(
        [
            _turn("", ("run", "bash_exec", '{"command": "false"}')),
            _SUBMIT_TURN,
        ]
    )
    env = _FakeEnvironment()
    with patch.object(
        env,
        "run_command",
        side_effect=RuntimeError("command failed") if raises else None,
        return_value=(1, "Error: bash_exec requires 'command' argument"),
    ) as run:
        result = _loop(
            model,
            ToolRegistry([BashExecTool(), SubmitTool()]),
            env,
            max_tool_calls=1,
            max_turns=3,
        ).run(_TASK)
    run.assert_called_once_with("false")
    assert result.final_turn_offered
    assert "[0 tool-calls remaining]" in result.tool_responses[0]["response_truncated"]
    assert "error" not in result.tool_calls[0]
    assert result.stopped_reason == StoppedReason.SUBMITTED


@pytest.mark.parametrize("caller", ["preflight", "loop"])
@pytest.mark.parametrize("nested", [False, True], ids=["kwargs", "extra_body"])
@pytest.mark.parametrize(
    "key",
    [
        "model",
        "messages",
        "tools",
        "tool_choice",
        "client",
        "max_attempts",
        "retry_sleep_secs",
    ],
)
def test_reserved_generation_fields_fail_before_any_request(
    caller: str, nested: bool, key: str
) -> None:
    handler = Mock()
    env = _FakeEnvironment()
    params = {key: "private-conflict-value"}
    generation = GenParams(kwargs={"extra_body": params} if nested else params)
    with _model(handler) as model, pytest.raises(ValueError) as error:
        if caller == "preflight":
            _check_tool_calling(model, generation)
        else:
            ReActLoop(
                model.client,
                "test",
                env,
                ToolRegistry([SubmitTool()]),
                generation_params=generation.as_request_params(),
            )
    assert f"reserved fields: {key}." in str(error.value)
    assert "generation.kwargs" in str(error.value)
    assert "private-conflict-value" not in str(error.value)
    assert "endpoint" not in str(error.value)
    assert not env.setup_called
    handler.assert_not_called()


@pytest.mark.parametrize("caller", ["preflight", "loop"])
@pytest.mark.parametrize(
    "params",
    [
        {"stream": True},
        {"extra_body": {"stream": True}},
        {"stream": False, "extra_body": {"stream": True}},
        {"stream": True, "extra_body": {"stream": False}},
    ],
)
def test_streaming_is_rejected_before_setup_or_any_request(caller, params) -> None:
    handler = Mock()
    env = _FakeEnvironment()
    generation = GenParams(kwargs=params)
    with _model(handler) as model, pytest.raises(ValueError, match="stream") as error:
        if caller == "preflight":
            _check_tool_calling(model, generation)
        else:
            ReActLoop(
                model.client,
                "test",
                env,
                ToolRegistry([SubmitTool()]),
                generation_params=generation.as_request_params(),
            )
    assert "generation.kwargs" in str(error.value)
    assert "endpoint" not in str(error.value)
    assert not env.setup_called
    handler.assert_not_called()


def test_allowed_generation_options_reach_preflight_and_loop() -> None:
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    generation = GenParams(
        temperature=0.3,
        top_p=0.9,
        max_completion_tokens=128,
        kwargs={
            "seed": 7,
            "parallel_tool_calls": False,
            "stream": False,
            "extra_body": {"provider_extension": {"enabled": True}, "stream": False},
        },
    )
    with _model(handler) as model:
        _check_tool_calling(model, generation)
        result = ReActLoop(
            model.client,
            "test",
            _FakeEnvironment(),
            ToolRegistry([SubmitTool()]),
            generation_params=generation.as_request_params(),
        ).run(_TASK)
    assert result.stopped_reason == StoppedReason.SUBMITTED
    assert len(requests) == 2
    assert [request["tool_choice"] for request in requests] == ["required", "auto"]
    assert requests[1]["max_completion_tokens"] == 128
    for request in requests:
        assert request["stream"] is False
        assert request["provider_extension"] == {"enabled": True}
        assert {
            key: request[key]
            for key in (
                "temperature",
                "top_p",
                "seed",
                "parallel_tool_calls",
            )
        } == {"temperature": 0.3, "top_p": 0.9, "seed": 7, "parallel_tool_calls": False}
