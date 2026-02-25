# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""SageMaker API model implementation of a model backend."""

import logging
import os
from functools import partial
from typing import Any, Iterable, cast

import boto3
import orjson
from botocore.config import Config
from tqdm import tqdm

from faith._internal.functools.retriable import RetryFunctionWrapper
from faith._internal.iter.fork_merge import ForkAndMergeTransform
from faith._internal.parsing.expr import evaluate_expr
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import (
    BaseModel,
    ChatResponse,
    GenerationError,
    PromptList,
    ReasoningSpec,
    _is_message_list,
)

logger = logging.getLogger(__name__)


_DEFAULT_REQUEST_BODY_EXPR = """
{
    "messages": messages,
} | {
    ("max_tokens" if k == "max_completion_tokens" else k): v
    for k, v in gen_params.items()
}
"""

_DEFAULT_RESPONSE_PARSING_EXPR = """
{
    "num_prompt_tokens": response_body["usage"]["prompt_tokens"],
    "output_text": response_body["choices"][0]["message"]["content"],
    "num_output_tokens": response_body["usage"]["completion_tokens"],
    "max_token_halt": response_body["choices"][0].get("finish_reason", None) == "length",
}
"""


class SageMakerModel(BaseModel):
    """A model that uses the SageMaker API to generate responses from a SageMaker endpoint."""

    def __init__(
        self,
        name_or_path: str,
        *,
        num_log_probs: int | None = None,
        reasoning_spec: ReasoningSpec | None = None,
        api_num_threads: int = 5,
        api_max_attempts: int = 10,
        api_retry_sleep_secs: float = 1.0,
        aws_region: str | None = None,
        endpoint_timeout_secs: int = 60,
        inference_component_name: str | None = None,
        request_body_expr: str = _DEFAULT_REQUEST_BODY_EXPR,
        response_parsing_expr: str = _DEFAULT_RESPONSE_PARSING_EXPR,
        **_kwargs: dict[str, Any],
    ):
        """Initialize the SageMaker API backend with the given parameters.

        Args:
            name_or_path: The model name or path (used as endpoint name).
            num_log_probs: Number of log probabilities to return (not supported).
            reasoning_spec: Specification of reasoning tokens (not supported).
            api_num_threads: Number of concurrent API threads.
            api_max_attempts: Maximum number of retry attempts.
            api_retry_sleep_secs: Sleep duration between retries.
            aws_region: AWS region for the SageMaker endpoint.
        """
        super().__init__(name_or_path)
        assert num_log_probs is None, "Logprobs are not supported for SageMaker models."
        assert (
            reasoning_spec is None
        ), "Reasoning tokens are not supported for SageMaker models."
        assert api_num_threads > 0, "Number of API threads must be greater than 0."
        assert api_max_attempts > 0, "Number of API attempts must be greater than 0."
        assert api_retry_sleep_secs > 0, "Retry sleep seconds must be greater than 0."
        self._api_num_threads = api_num_threads
        self._api_max_attempts = api_max_attempts
        self._api_retry_sleep_secs = api_retry_sleep_secs

        # Determine AWS region
        region = aws_region or os.getenv("AWS_REGION", "")
        assert (
            region
        ), "AWS region must be specified via 'aws_region' engine parameter or AWS_REGION environment variable."
        logger.info("Model %s in AWS region: %s", self._name_or_path, region)

        self._inference_component_name = inference_component_name
        self._request_body_expr = partial(evaluate_expr, request_body_expr)
        self._response_parsing_expr = partial(evaluate_expr, response_parsing_expr)

        # Initialize SageMaker runtime client
        self._client = boto3.client(
            "sagemaker-runtime",
            region_name=region,
            config=Config(read_timeout=endpoint_timeout_secs),
        )

    @property
    def supported_formats(self) -> set[PromptFormatter]:
        """Return the supported input formats for the SageMaker-based models."""
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
                SageMakerModel._handle_query_error,
                max_workers=self._api_num_threads,
            ),
            total=len(inputs),
            leave=False,
            desc="SageMaker Queries",
            unit=" prompts",
        )

    def _query_api(
        self, messages: list[dict[str, str]], **gen_params: Any
    ) -> ChatResponse:
        """Helper function to call the SageMaker API for a single message.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **gen_params: Additional generation parameters.

        Returns:
            ChatResponse object with the model's response.
        """
        request_kwargs = {
            "EndpointName": self._name_or_path,
            "ContentType": "application/json",
            "Accept": "application/json",
            "Body": orjson.dumps(
                self._request_body_expr(
                    names={"messages": messages, "gen_params": gen_params}
                )
            ),
        } | (
            {"InferenceComponentName": self._inference_component_name}
            if self._inference_component_name
            else {}
        )
        response = self._client.invoke_endpoint(**request_kwargs)

        # Parse the generated text and statistics from the response.
        response_body = orjson.loads(response["Body"].read())
        response = self._response_parsing_expr(names={"response_body": response_body})

        return ChatResponse(
            prompt_token_ids=None,  # Currently not specified in the response.
            num_prompt_tokens=response.get("num_prompt_tokens", None),
            prompt_text=None,  # Currently not specified in the response.
            output_token_ids=None,  # Currently not specified in the response.
            num_output_tokens=response.get("num_output_tokens", None),
            output_text=response.get("output_text", ""),
            request_token_ids=None,  # Currently not specified in the response.
            num_request_tokens=None,
            request_text=None,  # Currently not specified in the response.
            response_token_ids=None,  # Currently not specified in the response.
            num_response_tokens=None,
            response_text=response.get("output_text", ""),
            answer_token_ids=None,  # Currently not specified in the response.
            num_answer_tokens=None,
            answer_text=response.get("output_text", ""),
            max_token_halt=response.get("max_token_halt", False),
        )

    @staticmethod
    def _handle_query_error(exception: BaseException) -> GenerationError:
        """Handle exceptions raised when calling the API and return a GenerationError."""
        return GenerationError(title="SageMaker API Error", details=str(exception))
