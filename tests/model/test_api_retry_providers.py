# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""The other API engines also have only one retry owner."""

# pylint: disable=protected-access

import json
from collections.abc import Iterator
from io import BytesIO
from unittest.mock import Mock, patch

import boto3
import httpx
import pytest
from botocore.awsrequest import AWSResponse
from openrouter import OpenRouter
from urllib3.exceptions import ProtocolError

from faith._types.record.model_response import ChatResponse, GenerationError
from faith.model.open_router import OpenRouterModel
from faith.model.sagemaker import SageMakerModel


@pytest.mark.parametrize("status,count", [(400, 1), (503, 3)])
def test_openrouter_wire_budget(status: int, count: int) -> None:
    handler = Mock(
        return_value=httpx.Response(
            status,
            json={
                "error": {"code": status, "message": "synthetic failure"},
            },
        )
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        with patch(
            "faith.model.open_router.OpenRouter",
            side_effect=lambda **kwargs: OpenRouter(client=transport, **kwargs),
        ):
            model = OpenRouterModel("test", api_max_attempts=3, api_retry_sleep_secs=0)
        with patch("time.sleep") as sleep:
            result = list(model.query([[{"role": "user", "content": "test"}]]))[0]
    assert isinstance(result, GenerationError)
    assert result.details is not None
    assert f"HTTP {status}" in result.details
    assert handler.call_count == count
    sleep.assert_not_called()


@pytest.fixture(name="sagemaker_model")
def _sagemaker_model() -> Iterator[SageMakerModel]:
    create_client = boto3.client
    with patch(
        "faith.model.sagemaker.boto3.client",
        side_effect=lambda *args, **kwargs: create_client(
            *args,
            aws_access_key_id="test-only",
            aws_secret_access_key="test-only",
            **kwargs,
        ),
    ):
        model = SageMakerModel(
            "test",
            aws_region="us-east-1",
            api_max_attempts=3,
            api_retry_sleep_secs=0.25,
        )
    try:
        yield model
    finally:
        model._client.close()


def _sagemaker_response(
    status: int,
    body: bytes,
    *,
    interrupted: bool = False,
    content_length: int | None = None,
    headers: dict[str, str] | None = None,
) -> AWSResponse:
    raw = Mock()
    raw.stream.side_effect = lambda: iter([body])
    raw.read.side_effect = (
        ProtocolError("Connection broken while receiving response")
        if interrupted
        else BytesIO(body).read
    )
    return AWSResponse(
        "https://offline.invalid",
        status,
        {
            "content-type": "application/json",
            "content-length": str(
                len(body) if content_length is None else content_length
            ),
            **(headers or {}),
        },
        raw,
    )


@pytest.mark.parametrize(
    "status,body,count",
    [
        (400, b'{"__type":"ValidationException","Message":"synthetic"}', 1),
        (400, b'{"__type":"ThrottlingException","Message":"synthetic"}', 3),
        (503, b'{"__type":"ServiceUnavailable","Message":"synthetic"}', 3),
    ],
)
def test_sagemaker_wire_budget(
    sagemaker_model: SageMakerModel, status: int, body: bytes, count: int
) -> None:
    with (
        patch.object(
            sagemaker_model._client._endpoint.http_session,
            "send",
            return_value=_sagemaker_response(status, body),
        ) as send,
        patch("time.sleep") as sleep,
    ):
        result = list(sagemaker_model.query([[{"role": "user", "content": "test"}]]))[0]
    assert isinstance(result, GenerationError)
    assert result.details is not None
    assert f"HTTP {status}" in result.details
    assert send.call_count == count
    assert [call.args for call in sleep.call_args_list] == [(0.25,)] * (count - 1)


_SAGEMAKER_COMPLETION = json.dumps(
    {
        "choices": [{"message": {"content": "recovered"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }
).encode()


def _model_error_body(original_status: int | None) -> bytes:
    return json.dumps(
        {
            "__type": "ModelError",
            "Message": "private prompt test-only-secret",
            "OriginalStatusCode": original_status,
        }
    ).encode()


def _temporary_sagemaker_response(failure: str) -> AWSResponse:
    if failure.startswith("model_"):
        return _sagemaker_response(424, _model_error_body(int(failure.split("_")[1])))
    if failure == "stream_interruption":
        return _sagemaker_response(200, _SAGEMAKER_COMPLETION, interrupted=True)
    if failure == "incomplete_body":
        return _sagemaker_response(
            200, _SAGEMAKER_COMPLETION[:10], content_length=len(_SAGEMAKER_COMPLETION)
        )
    assert failure == "empty_choices"
    return _sagemaker_response(200, b'{"choices": []}')


@pytest.mark.parametrize(
    "failure,error_type",
    [
        ("model_429", "ModelError"),
        ("model_503", "ModelError"),
        ("stream_interruption", "ResponseStreamingError"),
        ("incomplete_body", "IncompleteReadError"),
        ("empty_choices", "EmptyChoicesError"),
    ],
)
@pytest.mark.parametrize("recovers", [True, False], ids=["recovery", "exhaustion"])
def test_sagemaker_temporary_failures(
    sagemaker_model: SageMakerModel,
    failure: str,
    error_type: str,
    recovers: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses = [_temporary_sagemaker_response(failure) for _ in range(3)]
    if recovers:
        responses[1] = _sagemaker_response(200, _SAGEMAKER_COMPLETION)
    with (
        patch.object(
            sagemaker_model._client._endpoint.http_session,
            "send",
            side_effect=responses,
        ) as send,
        patch("time.sleep") as sleep,
    ):
        result = list(sagemaker_model.query([[{"role": "user", "content": "test"}]]))[0]
    if recovers:
        assert isinstance(result, ChatResponse)
        assert result.output_text == "recovered"
    else:
        assert isinstance(result, GenerationError)
        assert result.details is not None
        assert error_type in result.details
        assert "3 total attempts" in result.details
    count = 2 if recovers else 3
    assert send.call_count == count
    assert [call.args for call in sleep.call_args_list] == [(0.25,)] * (count - 1)
    assert "private prompt" not in str(result) + caplog.text
    assert "test-only-secret" not in str(result) + caplog.text


@pytest.mark.parametrize("original_status", [400, 401, 403, 422, 501, 505, None, True])
def test_sagemaker_permanent_or_unknown_model_errors_fail_immediately(
    sagemaker_model: SageMakerModel, original_status: int | None
) -> None:
    with (
        patch.object(
            sagemaker_model._client._endpoint.http_session,
            "send",
            return_value=_sagemaker_response(424, _model_error_body(original_status)),
        ) as send,
        patch("time.sleep") as sleep,
    ):
        result = list(sagemaker_model.query([[{"role": "user", "content": "test"}]]))[0]
    assert isinstance(result, GenerationError)
    assert result.details == "ModelError (HTTP 424) code=ModelError"
    send.assert_called_once()
    sleep.assert_not_called()


def test_sagemaker_model_error_honors_retry_refusal(
    sagemaker_model: SageMakerModel,
) -> None:
    with (
        patch.object(
            sagemaker_model._client._endpoint.http_session,
            "send",
            return_value=_sagemaker_response(
                424, _model_error_body(503), headers={"x-should-retry": "false"}
            ),
        ) as send,
        patch("time.sleep") as sleep,
    ):
        result = list(sagemaker_model.query([[{"role": "user", "content": "test"}]]))[0]
    assert isinstance(result, GenerationError)
    send.assert_called_once()
    sleep.assert_not_called()
