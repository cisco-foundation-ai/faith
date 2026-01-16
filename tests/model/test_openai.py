# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import ChatResponse, GenerationError
from faith.model.openai import OpenAIModel


@patch("faith.model.openai.OpenAI", spec=True)
def test_openai_model(mock_openai_client_class: Mock) -> None:
    # Initialize the OpenAI model with a mock API key
    model = OpenAIModel("fake_model")
    mock_openai_client_class.assert_called_once_with(api_key=None)

    # Check if the model is initialized correctly.
    assert model.name_or_path == "fake_model"
    assert model.supported_formats == {PromptFormatter.CHAT}

    # Test a simple generation call (mocked)
    mock_openai_instance = mock_openai_client_class.return_value
    mock_openai_instance.chat.completions.create.return_value = ChatCompletion(
        id="chatcmpl-1234567890",
        object="chat.completion",
        created=1234567890,
        model="fake_model",
        usage=CompletionUsage(prompt_tokens=2, completion_tokens=5, total_tokens=7),
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Bin dabei!",
                ),
                finish_reason="stop",
            ),
        ],
    )
    response = model.query(inputs=[[{"role": "user", "content": "Hallo?"}]])
    assert list(response) == [
        ChatResponse(
            prompt_token_ids=None,
            num_prompt_tokens=2,
            prompt_text=None,
            output_token_ids=None,
            num_output_tokens=5,
            output_text="Bin dabei!",
            request_token_ids=None,
            num_request_tokens=2,
            request_text=None,
            response_token_ids=None,
            num_response_tokens=5,
            response_text="Bin dabei!",
            answer_token_ids=None,
            num_answer_tokens=5,
            answer_text="Bin dabei!",
            max_token_halt=False,
        )
    ]
    assert mock_openai_instance.chat.completions.create.call_count == 1


@patch("faith.model.openai.OpenAI", spec=True)
def test_openai_model_retry(mock_openai_client_class: Mock) -> None:
    # Initialize the OpenAI model with a mock API key
    model = OpenAIModel("fake_model", api_max_attempts=2, api_retry_sleep_secs=0.001)
    mock_openai_instance = mock_openai_client_class.return_value

    # Mock the API to raise an exception on the first call and succeed on the second
    mock_openai_instance.chat.completions.create.side_effect = [
        Exception("API error"),
        ChatCompletion(
            id="chatcmpl-1234567890",
            object="chat.completion",
            created=1234567890,
            model="fake_model",
            usage=CompletionUsage(prompt_tokens=2, completion_tokens=5, total_tokens=7),
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Bin dabei!",
                    ),
                    finish_reason="stop",
                ),
            ],
        ),
    ]

    response = model.query(inputs=[[{"role": "user", "content": "Hallo?"}]])

    assert list(response) == [
        ChatResponse(
            prompt_token_ids=None,
            num_prompt_tokens=2,
            prompt_text=None,
            output_token_ids=None,
            num_output_tokens=5,
            output_text="Bin dabei!",
            request_token_ids=None,
            num_request_tokens=2,
            request_text=None,
            response_token_ids=None,
            num_response_tokens=5,
            response_text="Bin dabei!",
            answer_token_ids=None,
            num_answer_tokens=5,
            answer_text="Bin dabei!",
            max_token_halt=False,
        )
    ]
    assert mock_openai_instance.chat.completions.create.call_count == 2


@patch("faith.model.openai.OpenAI", spec=True)
def test_openai_model_failed_call(mock_openai_client_class: Mock) -> None:
    # Initialize the OpenAI model with a mock API key
    model = OpenAIModel("fake_model", api_max_attempts=2, api_retry_sleep_secs=0.001)
    mock_openai_instance = mock_openai_client_class.return_value

    # Mock the API to raise an exception on the first call and succeed on the second
    mock_openai_instance.chat.completions.create.side_effect = [
        ValueError("Invalid request"),
        ValueError("Invalid request"),
    ]

    response = model.query(inputs=[[{"role": "user", "content": "Hallo?"}]])

    assert list(response) == [
        GenerationError(
            title="OpenAI API Error",
            details="Maximum retries exceeded...\nFinal exception: Invalid request",
        )
    ]
    assert mock_openai_instance.chat.completions.create.call_count == 2
