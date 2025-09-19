# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""A module for managing function execution allowing retries.

This module provides a RetryFunctionWrapper that can be used to wrap a function
and automatically retry its execution if it raises an exception. The number of
retries and the sleep duration between retries can be configured. If the maximum number
of retries is exceeded, a MaxAttemptsExceededError is raised.

The RetryFunctionWrapper can be used to make function calls more robust to
transient failures, such as network issues or temporary unavailability of resources.
This is particularly useful for calling external APIs or services that may
temporarily fail but are expected to succeed upon retrying eventually.
"""
import logging
import time
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

_OUT = TypeVar("_OUT")


class MaxAttemptsExceededError(Exception):
    """An exception raised when the maximum number of execution retries is exceeded."""


class RetryFunctionWrapper(Generic[_OUT]):
    """A wrapper for a function that retries its execution on failure."""

    def __init__(
        self, func: Callable[..., _OUT], max_attempts: int, retry_sleep_secs: float
    ) -> None:
        """Initialize with a function to apply and retry parameters."""
        assert retry_sleep_secs >= 0, "Retry sleep seconds must be non-negative."
        self._func = func
        self._max_attempts = max_attempts
        self._retry_sleep_secs = retry_sleep_secs

    def __call__(self, *args: Any, **kwargs: Any) -> _OUT:
        """Execute the function with retries on failure."""
        attempts = 0
        last_exp = None
        while attempts < self._max_attempts or self._max_attempts < 0:
            try:
                return self._func(*args, **kwargs)
            # pylint: disable=broad-exception-caught
            except Exception as e:
                logger.error("Exception while executing async call: %s", str(e))
                time.sleep(self._retry_sleep_secs)
                attempts += 1
                last_exp = e
        raise MaxAttemptsExceededError(
            f"Maximum retries exceeded...\nFinal exception: {str(last_exp)}"
        ) from last_exp
