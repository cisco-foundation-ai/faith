# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from unittest.mock import Mock, patch

import orjson

from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import ChatResponse, GenerationError
from faith.model.sagemaker import SageMakerModel


@patch("faith.model.sagemaker.boto3.client")
def test_sagemaker_model(mock_boto3_client: Mock) -> None:
    # Initialize the SageMaker model with a mock endpoint.
    model = SageMakerModel("fake_endpoint", aws_region="us-east-1")
    mock_boto3_client.assert_called_once()

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
            prompt_token_ids=None,
            num_prompt_tokens=2,
            prompt_text=None,
            output_token_ids=None,
            num_output_tokens=5,
            output_text="ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!",
            request_token_ids=None,
            num_request_tokens=None,
            request_text=None,
            response_token_ids=None,
            num_response_tokens=None,
            response_text="ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!",
            answer_token_ids=None,
            num_answer_tokens=None,
            answer_text="ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!",
            max_token_halt=False,
        )
    ]
    assert mock_sagemaker_client.invoke_endpoint.call_count == 1


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
        Exception("API error"),
        {"Body": BytesIO(orjson.dumps(mock_response_body))},
    ]

    response = model.query(inputs=[[{"role": "user", "content": "Hallo?"}]])

    assert list(response) == [
        ChatResponse(
            prompt_token_ids=None,
            num_prompt_tokens=2,
            prompt_text=None,
            output_token_ids=None,
            num_output_tokens=5,
            output_text="ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!",
            request_token_ids=None,
            num_request_tokens=None,
            request_text=None,
            response_token_ids=None,
            num_response_tokens=None,
            response_text="ನಾನು ಅಲ್ಲಿದ್ದೇನೆ!",
            answer_token_ids=None,
            num_answer_tokens=None,
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
            details="Maximum retries exceeded...\nFinal exception: Invalid request",
        )
    ]
    assert mock_sagemaker_client.invoke_endpoint.call_count == 3
