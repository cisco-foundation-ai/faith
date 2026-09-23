# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any
from unittest.mock import Mock, patch

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from faith._types.model.prompt import PromptFormatter
from faith._types.record.model_response import ChatResponse, GenerationError
from faith.model.openai import OpenAIModel

# A self-hosted OpenAI-compatible server, e.g. a local vLLM host.
_LOCAL_URL = "http://localhost:8000/v1"


@patch("faith.model.openai.OpenAI", spec=True)
def test_openai_model(mock_openai_client_class: Mock) -> None:
    model = OpenAIModel("fake_model")

    # Check if the model is initialized correctly.
    assert model.name_or_path == "fake_model"
    assert model.supported_formats == {PromptFormatter.CHAT}

    # Test a simple generation call (mocked).
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
            output_text="Bin dabei!",
            num_output_tokens=5,
            num_prompt_tokens=2,
            num_request_tokens=2,
            response_text="Bin dabei!",
            num_response_tokens=5,
            answer_text="Bin dabei!",
            num_answer_tokens=5,
            max_token_halt=False,
        )
    ]
    assert mock_openai_instance.chat.completions.create.call_count == 1


@pytest.mark.parametrize(
    ("env", "kwargs", "expected_api_key", "expected_base_url"),
    [
        # Real OpenAI with no key configured: stay None so the SDK fails fast at
        # construction with a clear error rather than an opaque 401 mid-run.
        ({}, {}, None, None),
        ({"OPENAI_API_KEY": "sk-from-env"}, {}, "sk-from-env", None),
        # An explicit api_key overrides the env var.
        (
            {"OPENAI_API_KEY": "sk-from-env"},
            {"api_key": "sk-local"},
            "sk-local",
            None,
        ),
        (
            {},
            {"base_url": _LOCAL_URL, "api_key": "sk-local"},
            "sk-local",
            _LOCAL_URL,
        ),
        # A keyless self-hosted server still needs a non-empty key, since the SDK
        # rejects None/""; "EMPTY" is the value keyless vLLM servers ignore.
        ({}, {"base_url": _LOCAL_URL}, "EMPTY", _LOCAL_URL),
        # An empty env var is no key at all, not a key of "".
        ({"OPENAI_API_KEY": ""}, {"base_url": _LOCAL_URL}, "EMPTY", _LOCAL_URL),
        ({"OPENAI_API_KEY": ""}, {}, None, None),
    ],
    ids=[
        "openai_keyless",
        "openai_env_key",
        "explicit_key_overrides_env",
        "custom_endpoint_with_key",
        "custom_endpoint_keyless",
        "custom_endpoint_empty_env_key",
        "openai_empty_env_key",
    ],
)
@patch("faith.model.openai.OpenAI", spec=True)
def test_openai_client_credentials(
    mock_openai_client_class: Mock,
    env: dict[str, str],
    kwargs: dict[str, Any],
    expected_api_key: str | None,
    expected_base_url: str | None,
) -> None:
    """How the api_key and base_url reaching the SDK are resolved.

    `base_url` points the client at an OpenAI-compatible server other than
    OpenAI's own (a vLLM host, OpenRouter, a local server), which is how agentic
    runs reach a self-hosted model.
    """
    with patch.dict(os.environ, env, clear=True):
        OpenAIModel("fake_model", **kwargs)
    mock_openai_client_class.assert_called_once_with(
        api_key=expected_api_key, base_url=expected_base_url
    )


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
            output_text="Bin dabei!",
            num_output_tokens=5,
            num_prompt_tokens=2,
            num_request_tokens=2,
            response_text="Bin dabei!",
            num_response_tokens=5,
            answer_text="Bin dabei!",
            num_answer_tokens=5,
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
