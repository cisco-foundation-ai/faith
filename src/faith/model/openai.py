# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""OpenAI API model implementation of a model backend."""

import os
from typing import Any

from openai import OpenAI

from faith.model.api_model import APIBasedModel
from faith.model.base import ChatResponse, GenerationError, ReasoningSpec


class OpenAIModel(APIBasedModel):
    """A model that uses the OpenAI API to generate responses from an OpenAI model."""

    def __init__(
        self,
        name_or_path: str,
        num_log_probs: int | None = None,
        reasoning_spec: ReasoningSpec | None = None,
        api_num_threads: int = 5,
        api_max_attempts: int = 10,
        api_retry_sleep_secs: float = 1.0,
        **_kwargs: dict[str, Any],
    ):
        """Initialize the OpenAI API backend with the given parameters."""
        super().__init__(
            name_or_path,
            num_log_probs=num_log_probs,
            reasoning_spec=reasoning_spec,
            api_num_threads=api_num_threads,
            api_max_attempts=api_max_attempts,
            api_retry_sleep_secs=api_retry_sleep_secs,
        )
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _query_api(
        self, messages: list[dict[str, str]], **gen_params: Any
    ) -> ChatResponse:
        """Helper function to call the OpenAI API for a single message."""
        response = self._client.chat.completions.create(
            model=self.name_or_path, messages=messages, **gen_params
        )
        # Note: output, response, and answer are the same for chat completions in OpenAI
        # since there is way to distinguish them in the response.
        return ChatResponse(
            prompt_token_ids=None,  # OpenAI does not return token IDs
            num_prompt_tokens=response.usage.prompt_tokens,
            prompt_text=None,  # OpenAI does not return prompt text
            output_token_ids=None,  # OpenAI does not return token IDs
            num_output_tokens=response.usage.completion_tokens,
            output_text=response.choices[0].message.content,
            request_token_ids=None,  # OpenAI does not return token IDs
            num_request_tokens=response.usage.prompt_tokens,
            request_text=None,  # OpenAI does not return prompt text
            response_token_ids=None,  # OpenAI does not return token IDs
            num_response_tokens=response.usage.completion_tokens,
            response_text=response.choices[0].message.content,
            answer_token_ids=None,  # OpenAI does not return token IDs
            num_answer_tokens=response.usage.completion_tokens,
            answer_text=response.choices[0].message.content,
            max_token_halt=response.choices[0].finish_reason == "length",
        )

    @staticmethod
    def _handle_query_error(exception: BaseException) -> GenerationError:
        """Handle exceptions raised when calling the API and return a GenerationError."""
        return GenerationError(title="OpenAI API Error", details=str(exception))
