# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Retry counts at the real SDK's HTTP transport, without external requests."""

import json
import ssl
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import OpenAI

from faith._types.model.generation import GenParams
from faith._types.record.agent import StoppedReason
from faith._types.record.model_response import GenerationError
from faith.agentic.core import Environment, ReActLoop, Task, ToolRegistry
from faith.agentic.tools.submit import SubmitTool
from faith.cli.subcmd.query import _check_tool_calling
from faith.model.api_retry import describe_api_error, is_retryable_api_error
from faith.model.openai import OpenAIModel


class _Environment(Environment):
    def setup(self, task: Task) -> None:
        pass

    def teardown(self) -> None:
        pass


def _completion(*, empty: bool = False, submit: bool = True) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": "done"}
    if submit:
        message["tool_calls"] = [
            {
                "id": "done",
                "type": "function",
                "function": {"name": "submit", "arguments": '{"answer":"done"}'},
            }
        ]
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        "choices": (
            []
            if empty
            else [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if submit else "stop",
                    "message": message,
                }
            ]
        ),
    }


@contextmanager
def _model(handler: Any, attempts: int | None = 3) -> Iterator[OpenAIModel]:
    kwargs: dict[str, Any] = {} if attempts is None else {"api_max_attempts": attempts}
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:

        def client(**options: Any) -> OpenAI:
            return OpenAI(http_client=transport, **options)

        with patch("faith.model.openai.OpenAI", side_effect=client):
            model = OpenAIModel(
                "test",
                api_key="test-only-secret",
                base_url="https://offline.invalid/v1",
                api_retry_sleep_secs=0.25,
                **kwargs,
            )
        yield model


def _invoke(model: OpenAIModel, caller: str) -> tuple[bool, str]:
    if caller == "qa":
        result = list(
            model.query(
                [[{"role": "user", "content": "private prompt"}]],
                temperature=0.7,
                max_completion_tokens=19,
            )
        )[0]
        return not isinstance(result, GenerationError), getattr(result, "details", "")
    if caller == "preflight":
        try:
            _check_tool_calling(
                model, GenParams(temperature=0.7, max_completion_tokens=19)
            )
        except ValueError as error:
            return False, str(error)
        return True, ""
    result = ReActLoop(
        model_api=model.client,
        model="test",
        environment=_Environment(),
        tool_registry=ToolRegistry([SubmitTool()]),
        api_max_attempts=model.api_max_attempts,
        api_retry_sleep_secs=model.api_retry_sleep_secs,
        generation_params={"temperature": 0.7, "max_completion_tokens": 19},
    ).run(Task(id="test", prompt="private prompt", ground_truth=["done"]))
    return result.stopped_reason == StoppedReason.SUBMITTED, result.error or ""


@pytest.mark.parametrize("caller", ["qa", "agentic", "preflight"])
@pytest.mark.parametrize("attempts", [3, None], ids=["configured-3", "default-10"])
@pytest.mark.parametrize(
    "status,retryable",
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (408, True),
        (409, True),
        (422, False),
        (429, True),
        (500, True),
        (501, False),
        (502, True),
        (503, True),
        (504, True),
        (505, False),
    ],
)
def test_persistent_http_failure_has_one_attempt_budget(
    caller: str,
    attempts: int | None,
    status: int,
    retryable: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            status,
            json={
                "error": {
                    "message": "private prompt test-only-secret",
                    "code": "synthetic_error",
                }
            },
        )

    with _model(handler, attempts) as model, patch("time.sleep") as sleep:
        ok, error = _invoke(model, caller)
    expected = (attempts or 10) if retryable else 1
    assert not ok
    assert len(requests) == expected
    assert sleep.call_count == expected - 1
    assert all(call.args == (0.25,) for call in sleep.call_args_list)
    assert f"HTTP {status}" in error
    assert "synthetic_error" in error
    assert "private prompt" not in error + caplog.text
    assert "test-only-secret" not in error + caplog.text
    assert all(request["temperature"] == 0.7 for request in requests)
    assert all(
        request["max_completion_tokens"] == (1024 if caller == "preflight" else 19)
        for request in requests
    )


@pytest.mark.parametrize("caller", ["qa", "agentic", "preflight"])
@pytest.mark.parametrize("failure", [429, 500, "empty", "timeout", "connection"])
def test_transient_failure_recovers_in_the_same_call(caller: str, failure: Any) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) <= 2:
            if failure == "timeout":
                raise httpx.ReadTimeout("private prompt", request=request)
            if failure == "connection":
                raise httpx.ConnectError("private prompt", request=request)
            if failure == "empty":
                return httpx.Response(200, json=_completion(empty=True))
            return httpx.Response(failure, json={"error": {"message": "temporary"}})
        return httpx.Response(200, json=_completion())

    with _model(handler) as model, patch("time.sleep") as sleep:
        assert _invoke(model, caller) == (True, "")
    assert len(requests) == 3
    assert requests[0] == requests[1] == requests[2]
    assert sleep.call_count == 2


@pytest.mark.parametrize("caller", ["qa", "agentic", "preflight"])
@pytest.mark.parametrize(
    "failure,expected", [("tls", 1), ("unknown", 1), ("timeout", 3), ("connection", 3)]
)
def test_transport_failures_are_classified_by_their_causes(
    caller: str, failure: str, expected: int
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if failure == "tls":
            raise httpx.ConnectError(
                "private prompt", request=request
            ) from ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED")
        if failure == "unknown":
            raise ValueError("private prompt test-only-secret")
        if failure == "timeout":
            raise httpx.ReadTimeout("private prompt", request=request)
        raise httpx.ConnectError("private prompt", request=request)

    with _model(handler) as model, patch("time.sleep") as sleep:
        ok, error = _invoke(model, caller)
    assert not ok
    assert calls == expected
    assert sleep.call_count == expected - 1
    assert "private prompt" not in error
    if failure == "tls":
        assert "SSLCertVerificationError (certificate verification failed)" in error
    if failure == "unknown":
        assert "ValueError" in error


@pytest.mark.parametrize(
    "headers,code,count,delays",
    [
        ({"retry-after": "2"}, None, 3, [2.0, 2.0]),
        ({"retry-after-ms": "500"}, None, 3, [0.5, 0.5]),
        ({"retry-after": "999999"}, None, 1, []),
        ({"retry-after": "NaN"}, None, 3, [0.25, 0.25]),
        ({"x-should-retry": "false"}, None, 1, []),
        ({}, "insufficient_quota", 1, []),
        ({}, "billing_hard_limit_reached", 1, []),
    ],
)
def test_server_delay_and_quota_policy(
    headers: dict[str, str], code: str | None, count: int, delays: list[float]
) -> None:
    handler = Mock(
        return_value=httpx.Response(
            429, headers=headers, json={"error": {"code": code}}
        )
    )
    with _model(handler) as model, patch("time.sleep") as sleep:
        assert not _invoke(model, "qa")[0]
    assert handler.call_count == count
    assert [call.args[0] for call in sleep.call_args_list] == delays


def test_direct_sdk_client_does_not_multiply_the_loop_budget() -> None:
    handler = Mock(
        return_value=httpx.Response(500, json={"error": {"message": "temporary"}})
    )
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as transport,
        OpenAI(
            api_key="test-only",
            http_client=transport,
            max_retries=2,
        ) as client,
        patch("time.sleep") as sleep,
    ):
        result = ReActLoop(
            model_api=client,
            model="test",
            environment=_Environment(),
            tool_registry=ToolRegistry(),
            api_max_attempts=3,
            api_retry_sleep_secs=0.25,
        ).run(Task(id="test", prompt="test", ground_truth=[]))
        assert client.max_retries == 2
    assert result.stopped_reason == StoppedReason.ERROR
    assert handler.call_count == 3
    assert sleep.call_count == 2


def test_late_model_failure_preserves_prior_messages_and_error() -> None:
    handler = Mock(
        side_effect=[
            httpx.Response(200, json=_completion(submit=False)),
            httpx.Response(403, json={"error": {"code": "permission_denied"}}),
        ]
    )
    with _model(handler) as model, patch("time.sleep") as sleep:
        result = ReActLoop(
            model_api=model.client,
            model="test",
            environment=_Environment(),
            tool_registry=ToolRegistry(),
            api_max_attempts=3,
        ).run(Task(id="late", prompt="test", ground_truth=[]))
    assert result.stopped_reason == StoppedReason.ERROR
    assert result.error is not None
    assert "HTTP 403" in result.error
    assert any(message.get("content") == "done" for message in result.messages)
    assert handler.call_count == 2
    sleep.assert_not_called()


def test_unknown_exception_and_suppressed_context_are_not_retryable() -> None:
    error = ValueError("private prompt")
    error.__context__ = ConnectionError("temporary")
    error.__suppress_context__ = True
    assert not is_retryable_api_error(error)
    assert describe_api_error(error) == "ValueError"


def test_empty_choices_exhaust_the_same_budget() -> None:
    handler = Mock(return_value=httpx.Response(200, json=_completion(empty=True)))
    with _model(handler) as model, patch("time.sleep") as sleep:
        ok, error = _invoke(model, "agentic")
    assert not ok
    assert "EmptyChoicesError" in error
    assert "3 total attempts" in error
    assert handler.call_count == 3
    assert sleep.call_count == 2


def test_retry_after_http_date_is_respected() -> None:
    handler = Mock(
        return_value=httpx.Response(
            429,
            headers={"retry-after": "Thu, 01 Jan 1970 00:00:02 GMT"},
            json={"error": {"message": "temporary"}},
        )
    )
    with (
        _model(handler) as model,
        patch("time.time", return_value=0),
        patch("time.sleep") as sleep,
    ):
        assert not _invoke(model, "qa")[0]
    assert [call.args for call in sleep.call_args_list] == [(2.0,), (2.0,)]
