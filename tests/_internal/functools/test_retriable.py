# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from functools import reduce

import pytest

from faith._internal.functools.retriable import (
    MaxAttemptsExceededError,
    RetryFunctionWrapper,
)


class FailingFunction:
    """A function that fails a specified number of times before succeeding."""

    def __init__(self, fail_times: int):
        self._fail_times = fail_times
        self._calls = 0

    def __call__(self, *args: str) -> str:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise ValueError("Simulated failure")
        return reduce(lambda x, y: x + y, args[: (self._calls - 1)], "")

    @property
    def calls(self) -> int:
        """Number of times the function has been called."""
        return self._calls


@pytest.mark.parametrize(
    "fail_times, expected",
    [
        (0, ""),
        (1, "a"),
        (2, "ab"),
        (3, "abc"),
        (4, "abcd"),
        (5, "abcde"),
    ],
)
def test_retry_function_wrapper_success(fail_times: int, expected: str) -> None:
    """Test that the function succeeds after the specified number of failures."""
    func = FailingFunction(fail_times)
    wrapper = RetryFunctionWrapper(func, max_attempts=6, retry_sleep_secs=0.001)
    result = wrapper("a", "b", "c", "d", "e")
    assert result == expected
    assert func.calls == fail_times + 1


def test_retry_function_wrapper_max_attempts_exceeded() -> None:
    """Test that the function raises an error when max attempts are exceeded."""
    func = FailingFunction(5)
    wrapper = RetryFunctionWrapper(func, max_attempts=5, retry_sleep_secs=0.001)
    with pytest.raises(
        MaxAttemptsExceededError,
        match="Maximum retries exceeded...\nFinal exception: Simulated failure",
    ):
        wrapper("a", "b", "c", "d", "e", "f")
    assert func.calls == 5
