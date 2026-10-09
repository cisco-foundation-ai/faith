# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from typing import Any
from unittest.mock import Mock, patch

import orjson
import pytest

from faith._types.model.prompt import PromptFormatter
from faith._types.record.model_response import ChatResponse, GenerationError
from faith.model.sagemaker import SageMakerModel


@patch("faith.model.sagemaker.boto3.client")
def test_sagemaker_model(mock_boto3_client: Mock) -> None:
    # Initialize the SageMaker model with a mock endpoint.
    model = SageMakerModel("fake_endpoint", aws_region="us-east-1")
    mock_boto3_client.assert_called_once()
    assert mock_boto3_client.call_args.kwargs["config"].retries == {
        "total_max_attempts": 1
    }

    # Check if the model is initialized correctly.
    assert model.name_or_path == "fake_endpoint"
    assert model.supported_formats == {PromptFormatter.CHAT}

    # Test a simple generation call (mocked).
    mock_sagemaker_client = mock_boto3_client.return_value
    mock_response_body = {
        "usage": {"prompt_tokens": 2, "completion_tokens": 5},
        "choices": [
            {
                "message": {"role": "assistant", "content": "ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!"},
                "finish_reason": "stop",
            }
        ],
    }
    mock_sagemaker_client.invoke_endpoint.return_value = {
        "Body": BytesIO(orjson.dumps(mock_response_body)),
    }

    response = model.query(inputs=[[{"role": "user", "content": "Hallo?"}]])
    assert list(response) == [
        ChatResponse(
            output_text="ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!",
            num_output_tokens=5,
            num_prompt_tokens=2,
            response_text="ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!",
            answer_text="ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!",
            max_token_halt=False,
        )
    ]
    assert mock_sagemaker_client.invoke_endpoint.call_count == 1


@pytest.mark.parametrize(
    "body,expression",
    [
        (
            {"choices": [], "generated_text": "custom output"},
            '{"output_text": response_body["generated_text"]}',
        ),
        (["custom output"], '{"output_text": response_body[0]}'),
    ],
)
@patch("faith.model.sagemaker.boto3.client")
def test_sagemaker_custom_response_parser(
    mock_boto3_client: Mock, body: Any, expression: str
) -> None:
    model = SageMakerModel(
        "fake_endpoint", aws_region="us-east-1", response_parsing_expr=expression
    )
    mock_boto3_client.return_value.invoke_endpoint.return_value = {
        "Body": BytesIO(orjson.dumps(body))
    }
    result = list(model.query([[{"role": "user", "content": "test"}]]))[0]
    assert isinstance(result, ChatResponse)
    assert result.output_text == "custom output"
    mock_boto3_client.return_value.invoke_endpoint.assert_called_once()


@patch("faith.model.sagemaker.boto3.client")
def test_sagemaker_custom_parser_index_error_is_not_retried(
    mock_boto3_client: Mock,
) -> None:
    model = SageMakerModel(
        "fake_endpoint",
        aws_region="us-east-1",
        response_parsing_expr='{"output_text": response_body["choices"][0]}',
    )
    mock_boto3_client.return_value.invoke_endpoint.return_value = {
        "Body": BytesIO(b'{"choices": []}')
    }
    with patch("time.sleep") as sleep:
        result = list(model.query([[{"role": "user", "content": "test"}]]))[0]
    assert isinstance(result, GenerationError)
    assert result.details == "IndexError"
    mock_boto3_client.return_value.invoke_endpoint.assert_called_once()
    sleep.assert_not_called()


@patch("faith.model.sagemaker.boto3.client")
def test_sagemaker_model_retry(mock_boto3_client: Mock) -> None:
    # Initialize the SageMaker model with retry settings.
    model = SageMakerModel(
        "fake_endpoint",
        aws_region="us-east-1",
        api_max_attempts=2,
        api_retry_sleep_secs=0.001,
    )
    mock_sagemaker_client = mock_boto3_client.return_value

    # Mock the API to raise an exception on the first call and succeed on the second.
    mock_response_body = {
        "usage": {"prompt_tokens": 2, "completion_tokens": 5},
        "choices": [
            {
                "message": {"role": "assistant", "content": "ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!"},
                "finish_reason": "stop",
            }
        ],
    }
    mock_sagemaker_client.invoke_endpoint.side_effect = [
        ConnectionError("API error"),
        {"Body": BytesIO(orjson.dumps(mock_response_body))},
    ]

    response = model.query(inputs=[[{"role": "user", "content": "Hallo?"}]])

    assert list(response) == [
        ChatResponse(
            output_text="ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!",
            num_output_tokens=5,
            num_prompt_tokens=2,
            response_text="ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!",
            answer_text="ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!",
            max_token_halt=False,
        )
    ]
    assert mock_sagemaker_client.invoke_endpoint.call_count == 2


@patch("faith.model.sagemaker.boto3.client")
def test_sagemaker_model_failed_call(mock_boto3_client: Mock) -> None:
    # Initialize the SageMaker model with retry settings.
    model = SageMakerModel(
        "fake_endpoint",
        aws_region="us-east-1",
        api_max_attempts=3,
        api_retry_sleep_secs=0.001,
    )
    mock_sagemaker_client = mock_boto3_client.return_value

    # Mock the API to raise an exception on all calls.
    mock_sagemaker_client.invoke_endpoint.side_effect = [
        ValueError("Invalid request"),
        ValueError("Invalid request"),
        ValueError("Invalid request"),
    ]

    response = model.query(inputs=[[{"role": "user", "content": "Hallo?"}]])

    assert list(response) == [
        GenerationError(
            title="SageMaker API Error",
            details="ValueError",
        )
    ]
    assert mock_sagemaker_client.invoke_endpoint.call_count == 1
