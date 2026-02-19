# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import ChatResponse, GenerationError
from faith.model.open_router import OpenRouterModel


@patch("faith.model.open_router.OpenRouter")
def test_open_router_model(mock_openrouter_client_class: Mock) -> None:
    # Initialize the OpenRouter model with a mock API key.
    model = OpenRouterModel("anthropic/claude-3.5-sonnet")
    mock_openrouter_client_class.assert_called_once_with(api_key="")

    # Check if the model is initialized correctly.
    assert model.name_or_path == "anthropic/claude-3.5-sonnet"
    assert model.supported_formats == {PromptFormatter.CHAT}

    # Test a simple generation call (mocked).
    mock_openrouter_instance = mock_openrouter_client_class.return_value
    mock_openrouter_instance.chat.send.return_value = Mock(
        choices=[Mock(message=Mock(content="Bin dabei!"), finish_reason="stop")],
        usage=Mock(prompt_tokens=2, completion_tokens=5),
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
    assert mock_openrouter_instance.chat.send.call_count == 1


@patch("faith.model.open_router.OpenRouter")
def test_open_router_model_retry(mock_openrouter_client_class: Mock) -> None:
    # Initialize the OpenRouter model with a mock API key
    model = OpenRouterModel(
        "anthropic/claude-3.5-sonnet",
        api_max_attempts=2,
        api_retry_sleep_secs=0.001,
    )
    mock_openrouter_instance = mock_openrouter_client_class.return_value

    # Mock the API to raise an exception on the first call and succeed on the second
    mock_openrouter_instance.chat.send.side_effect = [
        Exception("API error"),
        Mock(
            choices=[Mock(message=Mock(content="Bin dabei!"), finish_reason="stop")],
            usage=Mock(prompt_tokens=2, completion_tokens=5),
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
    assert mock_openrouter_instance.chat.send.call_count == 2


@patch("faith.model.open_router.OpenRouter")
def test_open_router_model_failed_call(mock_openrouter_client_class: Mock) -> None:
    # Initialize the OpenRouter model with a mock API key
    model = OpenRouterModel(
        "anthropic/claude-3.5-sonnet",
        api_max_attempts=2,
        api_retry_sleep_secs=0.001,
    )
    mock_openrouter_instance = mock_openrouter_client_class.return_value

    # Mock the API to raise an exception on all calls
    mock_openrouter_instance.chat.send.side_effect = [
        ValueError("Invalid request"),
        ValueError("Invalid request"),
    ]

    response = model.query(inputs=[[{"role": "user", "content": "Hallo?"}]])

    assert list(response) == [
        GenerationError(
            title="OpenRouter API Error",
            details="Maximum retries exceeded...\nFinal exception: Invalid request",
        )
    ]
    assert mock_openrouter_instance.chat.send.call_count == 2
