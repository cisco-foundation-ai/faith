# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""OpenAI API model implementation of a model backend."""
import os
from typing import Any, Iterable, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from faith._internal.functools.retriable import RetryFunctionWrapper
from faith._internal.iter.fork_merge import ForkAndMergeTransform
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import (
    BaseModel,
    ChatResponse,
    GenerationError,
    PromptList,
    _is_message_list,
)


class OpenAIModel(BaseModel):
    """A model that uses the OpenAI API to generate responses from an OpenAI model."""

    def __init__(
        self,
        name_or_path: str,
        num_log_probs: int | None = None,
        api_num_threads: int = 5,
        api_max_attempts: int = 10,
        api_retry_sleep_secs: float = 1.0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the OpenAI API backend with the given parameters."""
        super().__init__(name_or_path)
        assert num_log_probs is None, "Logprobs are not supported for OpenAI models."
        assert api_num_threads > 0, "Number of API threads must be greater than 0."
        assert api_max_attempts > 0, "Number of API attempts must be greater than 0."
        assert api_retry_sleep_secs > 0, "Retry sleep seconds must be greater than 0."
        self._api_num_threads = api_num_threads
        self._api_max_attempts = api_max_attempts
        self._api_retry_sleep_secs = api_retry_sleep_secs
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @property
    def supported_formats(self) -> set[PromptFormatter]:
        """Return the supported input formats for the OpenAI-based models."""
        return set([PromptFormatter.CHAT])

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

        if max_completion_tokens > 0:
            # See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-max_completion_tokens
            gen_params["max_completion_tokens"] = max_completion_tokens

        return cast(
            Iterable[ChatCompletionMessageParam], inputs
        ) >> ForkAndMergeTransform[
            ChatCompletionMessageParam, ChatResponse | GenerationError
        ](
            RetryFunctionWrapper[ChatResponse](
                lambda msg: self._query_api(msg, **gen_params),
                max_attempts=self._api_max_attempts,
                retry_sleep_secs=self._api_retry_sleep_secs,
            ),
            OpenAIModel._handle_query_error,
            max_workers=self._api_num_threads,
        )

    def _query_api(
        self, message: Iterable[ChatCompletionMessageParam], **gen_params: Any
    ) -> ChatResponse:
        """Helper function to call the OpenAI API for a single message."""
        response = self._client.chat.completions.create(
            model=self.name_or_path, messages=message, **gen_params
        )
        return ChatResponse(
            prompt_token_ids=None,  # OpenAI does not return token IDs
            num_prompt_tokens=response.usage.prompt_tokens,
            prompt_text=None,  # OpenAI does not return prompt text
            output_token_ids=None,  # OpenAI does not return token IDs
            num_output_tokens=response.usage.completion_tokens,
            output_text=response.choices[0].message.content,
            max_token_halt=response.choices[0].finish_reason == "length",
        )

    @staticmethod
    def _handle_query_error(exception: BaseException) -> GenerationError:
        """Handle exceptions raised when calling the API and return a GenerationError."""
        return GenerationError(title="OpenAI API Error", details=str(exception))
