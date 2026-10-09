# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Shared API retry policy; provider imports are optional and lazy."""

# pylint: disable=import-outside-toplevel

import math
import re
import ssl
import time
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Any

from faith._internal.functools.retriable import (
    MaxAttemptsExceededError,
    RetryFunctionWrapper,
    exception_chain,
)


class EmptyChoicesError(Exception):
    """The completion API returned an empty choices array."""


def _http_status(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode", status)
    if not isinstance(status, int) or isinstance(status, bool):
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _certificate_error(error: BaseException) -> bool:
    return isinstance(error, ssl.SSLCertVerificationError) or (
        isinstance(error, ssl.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(error)
    )


def _error_code(error: BaseException) -> str | None:
    code = getattr(error, "code", None)
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", code)
    return code if isinstance(code, str) else None


def _response_headers(error: BaseException) -> Any:
    response = getattr(error, "response", None)
    if response is None:
        response = getattr(error, "raw_response", None)
    if isinstance(response, dict):
        return response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    return getattr(response, "headers", {})


@lru_cache(maxsize=1)
def _transport_errors() -> tuple[type[Exception], ...]:
    errors: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)
    try:
        from httpx import NetworkError, RemoteProtocolError, TimeoutException

        errors += (NetworkError, RemoteProtocolError, TimeoutException)
    except ImportError:
        pass
    try:
        from botocore.exceptions import (
            ConnectionClosedError,
            ConnectTimeoutError,
            EndpointConnectionError,
            IncompleteReadError,
            ReadTimeoutError,
            ResponseStreamingError,
        )

        errors += (
            ConnectionClosedError,
            ConnectTimeoutError,
            EndpointConnectionError,
            IncompleteReadError,
            ReadTimeoutError,
            ResponseStreamingError,
        )
    except ImportError:
        pass
    return errors


def is_retryable_api_error(error: Exception) -> bool:
    """Retry known temporary failures, never certificate or unknown failures."""
    chain = tuple(exception_chain(error))
    if any(_certificate_error(cause) for cause in chain):
        return False
    for cause in chain:
        status = _http_status(cause)
        if status is not None:
            if _response_headers(cause).get("x-should-retry") == "false" or _error_code(
                cause
            ) in {"insufficient_quota", "billing_hard_limit_reached"}:
                return False
            response = getattr(cause, "response", None)
            if isinstance(response, dict):
                code = _error_code(cause)
                if code in {
                    "Throttling",
                    "ThrottlingException",
                    "ModelNotReadyException",
                    "RequestTimeout",
                    "RequestTimeoutException",
                }:
                    return True
                # SageMaker wraps inference-container failures in HTTP 424.
                if status == 424 and code == "ModelError":
                    original_status = response.get("OriginalStatusCode")
                    if isinstance(original_status, int) and not isinstance(
                        original_status, bool
                    ):
                        status = original_status
            return status in {408, 409, 429} or (
                500 <= status < 600 and status not in {501, 505}
            )
    return any(
        isinstance(cause, (EmptyChoicesError, *_transport_errors())) for cause in chain
    )


def api_retry_delay(error: Exception, default: float) -> float | None:
    """Honor a server delay up to 60 seconds; decline longer delays."""
    for cause in exception_chain(error):
        headers = _response_headers(cause)
        for key, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
            value = headers.get(key)
            if not isinstance(value, str):
                continue
            try:
                delay = float(value) / divisor
            except ValueError:
                try:
                    delay = parsedate_to_datetime(value).timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    continue
            if math.isfinite(delay) and delay >= 0:
                return max(default, delay) if delay <= 60 else None
    return default


def describe_api_error(error: BaseException) -> str:
    """Report cause types/status/identifiers, never bodies, URLs or headers."""
    parts = []
    for cause in exception_chain(error):
        if isinstance(cause, MaxAttemptsExceededError):
            parts.append(str(cause))
            continue
        detail = type(cause).__name__
        if _certificate_error(cause):
            detail += " (certificate verification failed)"
        if (status := _http_status(cause)) is not None:
            detail += f" (HTTP {status})"
        for name in ("code", "param"):
            value = _error_code(cause) if name == "code" else getattr(cause, name, None)
            if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
                detail += f" {name}={value}"
        parts.append(detail)
    return " caused by ".join(parts)


def without_sdk_retries(client: Any) -> Any:
    """Copy a real OpenAI client with SDK retries disabled; leave fakes alone."""
    try:
        from openai import OpenAI
    except ImportError:
        return client
    if isinstance(client, OpenAI):
        return client.with_options(max_retries=0)
    return client


def completion_with_retries(
    client: Any, *, max_attempts: int, retry_sleep_secs: float, **kwargs: Any
) -> Any:
    """Call a completion API with one retry owner and explicit empty recovery."""
    client = without_sdk_retries(client)

    def create() -> Any:
        response = client.chat.completions.create(**kwargs)
        if not response.choices:
            raise EmptyChoicesError()
        return response

    return RetryFunctionWrapper[Any](
        create,
        max_attempts,
        retry_sleep_secs,
        retry_if=is_retryable_api_error,
        retry_delay=api_retry_delay,
    )()
