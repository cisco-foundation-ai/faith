# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""OpenRouter API model implementation of a model backend."""

import os
from typing import Any, Iterable, cast

from openrouter import OpenRouter
from tqdm import tqdm

from faith._internal.functools.retriable import RetryFunctionWrapper
from faith._internal.iter.fork_merge import ForkAndMergeTransform
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import (
    BaseModel,
    ChatResponse,
    GenerationError,
    PromptList,
    ReasoningSpec,
    _is_message_list,
)


class OpenRouterModel(BaseModel):
    """A model that uses the OpenRouter API to generate responses."""

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
        """Initialize the OpenRouter API backend with the given parameters."""
        super().__init__(name_or_path)
        assert (
            num_log_probs is None
        ), "Logprobs are not supported for OpenRouter models."
        assert (
            reasoning_spec is None
        ), "Reasoning tokens are not supported for OpenRouter models."
        assert api_num_threads > 0, "Number of API threads must be greater than 0."
        assert api_max_attempts > 0, "Number of API attempts must be greater than 0."
        assert api_retry_sleep_secs > 0, "Retry sleep seconds must be greater than 0."
        self._api_num_threads = api_num_threads
        self._api_max_attempts = api_max_attempts
        self._api_retry_sleep_secs = api_retry_sleep_secs
        self._client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY", ""))

    @property
    def supported_formats(self) -> set[PromptFormatter]:
        """Return the supported input formats for the OpenRouter-based models."""
        return {PromptFormatter.CHAT}

    def query(
        self,
        inputs: PromptList,
        _verbose_resps: bool = False,
        max_completion_tokens: int = 500,
        **gen_params: Any,
    ) -> Iterable[ChatResponse | GenerationError]:
        """Map each input in `inputs` to the model's generated response for it."""
        assert _is_message_list(
            inputs
        ), "All inputs must be a list of messages. Please convert it to a list of messages before passing it to the model."

        # Note: max_completion_tokens may not be the name of this field for all models.
        if max_completion_tokens > 0:
            gen_params["max_completion_tokens"] = max_completion_tokens

        yield from tqdm(
            cast(Iterable[list[dict[str, str]]], inputs)
            >> ForkAndMergeTransform[
                list[dict[str, str]], ChatResponse | GenerationError
            ](
                RetryFunctionWrapper[ChatResponse](
                    lambda msg: self._query_api(msg, **gen_params),
                    max_attempts=self._api_max_attempts,
                    retry_sleep_secs=self._api_retry_sleep_secs,
                ),
                OpenRouterModel._handle_query_error,
                max_workers=self._api_num_threads,
            ),
            total=len(inputs),
            leave=False,
            desc="OpenRouter Queries",
            unit=" prompts",
        )

    def _query_api(
        self, messages: list[dict[str, str]], **gen_params: Any
    ) -> ChatResponse:
        """Helper function to call the OpenRouter API for a single message."""
        response = self._client.chat.send(
            model=self.name_or_path, messages=messages, **gen_params
        )
        return ChatResponse(
            prompt_token_ids=None,
            num_prompt_tokens=int(response.usage.prompt_tokens),
            prompt_text=None,
            output_token_ids=None,
            num_output_tokens=int(response.usage.completion_tokens),
            output_text=response.choices[0].message.content,
            request_token_ids=None,
            num_request_tokens=int(response.usage.prompt_tokens),
            request_text=None,
            response_token_ids=None,
            num_response_tokens=int(response.usage.completion_tokens),
            response_text=response.choices[0].message.content,
            answer_token_ids=None,
            num_answer_tokens=int(response.usage.completion_tokens),
            answer_text=response.choices[0].message.content,
            max_token_halt=response.choices[0].finish_reason == "length",
        )

    @staticmethod
    def _handle_query_error(exception: BaseException) -> GenerationError:
        """Handle exceptions raised when calling the API and return a GenerationError."""
        return GenerationError(title="OpenRouter API Error", details=str(exception))
