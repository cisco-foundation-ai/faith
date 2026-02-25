# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Common base class for API-based model backends."""

from abc import ABC, abstractmethod
from typing import Any, Iterable, cast

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


class APIBasedModel(BaseModel, ABC):
    """Base class for model backends that query an external API."""

    def __init__(
        self,
        name_or_path: str,
        num_log_probs: int | None = None,
        reasoning_spec: ReasoningSpec | None = None,
        api_num_threads: int = 5,
        api_max_attempts: int = 10,
        api_retry_sleep_secs: float = 1.0,
    ):
        """Initialize the API-based model backend with the given parameters."""
        super().__init__(name_or_path)
        assert num_log_probs is None, "Logprobs are not supported by API-based models."
        assert reasoning_spec is None, "Reasoning is not supported by API-based models."
        assert api_num_threads > 0, "Number of API threads must be greater than 0."
        assert api_max_attempts > 0, "Number of API attempts must be greater than 0."
        assert api_retry_sleep_secs > 0, "Retry sleep seconds must be greater than 0."
        self._api_num_threads = api_num_threads
        self._api_max_attempts = api_max_attempts
        self._api_retry_sleep_secs = api_retry_sleep_secs

    @property
    def supported_formats(self) -> set[PromptFormatter]:
        """Return the supported input formats for API-based models."""
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

        # Note: max_completion_tokens may not be the name of this field for all apis.
        # We may need to consider a more general solution in the future.
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
                self._handle_query_error,
                max_workers=self._api_num_threads,
            ),
            total=len(inputs),
            leave=False,
            desc="Model API Queries",
            unit=" prompts",
        )

    @abstractmethod
    def _query_api(
        self, messages: list[dict[str, str]], **gen_params: Any
    ) -> ChatResponse:
        """Call the API for a single message and return the response."""

    @staticmethod
    @abstractmethod
    def _handle_query_error(exception: BaseException) -> GenerationError:
        """Handle exceptions raised when calling the API and return a GenerationError."""
