# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""SageMaker API model implementation of a model backend."""

import logging
import os
from functools import partial
from typing import Any

import boto3
import orjson
from botocore.config import Config

from faith._internal.parsing.expr import evaluate_expr
from faith._types.model.spec import Reasoning
from faith._types.records.model_response import ChatResponse, GenerationError
from faith.model.api_model import APIBasedModel

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
    "max_token_halt": response_body["choices"][0].get("finish_reason") == "length",
}
"""


class SageMakerModel(APIBasedModel):
    """A model that uses the SageMaker API to generate responses from a SageMaker endpoint."""

    def __init__(
        self,
        name_or_path: str,
        *,
        num_log_probs: int | None = None,
        reasoning_spec: Reasoning | None = None,
        api_num_threads: int = 5,
        api_max_attempts: int = 10,
        api_retry_sleep_secs: float = 1.0,
        aws_region: str | None = None,
        endpoint_timeout_secs: int = 60,
        inference_component_name: str | None = None,
        request_body_expr: str = _DEFAULT_REQUEST_BODY_EXPR,
        response_parsing_expr: str = _DEFAULT_RESPONSE_PARSING_EXPR,
        **_kwargs: Any,
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
        super().__init__(
            name_or_path,
            num_log_probs=num_log_probs,
            reasoning_spec=reasoning_spec,
            api_num_threads=api_num_threads,
            api_max_attempts=api_max_attempts,
            api_retry_sleep_secs=api_retry_sleep_secs,
        )

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
            output_text=response.get("output_text") or "",
            num_output_tokens=response.get("num_output_tokens"),
            num_prompt_tokens=response.get("num_prompt_tokens"),
            response_text=response.get("output_text") or "",
            answer_text=response.get("output_text") or "",
            max_token_halt=response.get("max_token_halt") or False,
        )

    @staticmethod
    def _handle_query_error(exception: BaseException) -> GenerationError:
        """Handle exceptions raised when calling the API and return a GenerationError."""
        return GenerationError(title="SageMaker API Error", details=str(exception))
