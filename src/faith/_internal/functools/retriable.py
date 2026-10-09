# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded retries with caller-supplied failure classification."""

import logging
import math
import time
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

_OUT = TypeVar("_OUT")


class MaxAttemptsExceededError(Exception):
    """A retryable call exhausted its total attempt budget."""


def exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Walk explicit causes or unsuppressed contexts without cycling."""
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        yield error
        cause = error.__cause__
        if cause is None and not error.__suppress_context__:
            cause = error.__context__
        if cause is None:
            break
        error = cause


def validate_retry_settings(max_attempts: int, retry_sleep_secs: float) -> None:
    """Require a finite attempt budget and finite, non-negative delay."""
    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or max_attempts < 1
    ):
        raise ValueError("api_max_attempts must be an integer of at least 1.")
    if not math.isfinite(retry_sleep_secs) or retry_sleep_secs < 0:
        raise ValueError("api_retry_sleep_secs must be finite and non-negative.")


def _retry_connection_error(error: Exception) -> bool:
    return isinstance(error, (ConnectionError, TimeoutError))


class RetryFunctionWrapper(Generic[_OUT]):
    """Retry classified failures, counting the initial call as attempt one."""

    def __init__(
        self,
        func: Callable[..., _OUT],
        max_attempts: int,
        retry_sleep_secs: float,
        *,
        retry_if: Callable[[Exception], bool] = _retry_connection_error,
        retry_delay: Callable[[Exception, float], float | None] | None = None,
    ) -> None:
        """Initialize with a function to apply and retry parameters."""
        validate_retry_settings(max_attempts, retry_sleep_secs)
        self._func = func
        self._max_attempts = max_attempts
        self._retry_sleep_secs = retry_sleep_secs
        self._retry_if = retry_if
        self._retry_delay = retry_delay

    def __call__(self, *args: Any, **kwargs: Any) -> _OUT:
        """Execute the function, sleeping only before another permitted attempt."""
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._func(*args, **kwargs)
            # pylint: disable-next=broad-exception-caught
            except Exception as error:
                if not self._retry_if(error):
                    raise
                if attempt == self._max_attempts:
                    raise MaxAttemptsExceededError(
                        f"Maximum attempts exceeded ({attempt} total attempts)."
                    ) from error
                delay = (
                    self._retry_delay(error, self._retry_sleep_secs)
                    if self._retry_delay is not None
                    else self._retry_sleep_secs
                )
                if delay is None:
                    raise
                logger.warning(
                    "Retrying %s after attempt %d/%d in %s seconds",
                    type(error).__name__,
                    attempt,
                    self._max_attempts,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
        raise AssertionError("Unreachable with a positive attempt budget")
