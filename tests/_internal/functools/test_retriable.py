# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from unittest.mock import Mock, patch

import pytest

from faith._internal.functools.retriable import (
    MaxAttemptsExceededError,
    RetryFunctionWrapper,
    exception_chain,
)


class FailingFunction:
    """A function that fails a specified number of times before succeeding."""

    def __init__(self, fail_times: int):
        self._fail_times = fail_times
        self._calls = 0

    def __call__(self, *args: str) -> str:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise ConnectionError("Simulated failure")
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
        match=r"Maximum attempts exceeded \(5 total attempts\)",
    ):
        wrapper("a", "b", "c", "d", "e", "f")
    assert func.calls == 5


@pytest.mark.parametrize("attempts", [0, -1, 1.5, True])
def test_retry_budget_must_be_a_positive_integer(attempts: int) -> None:
    with pytest.raises(ValueError, match="api_max_attempts"):
        RetryFunctionWrapper(Mock(), attempts, 0)


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf")])
def test_retry_delay_must_be_finite_and_non_negative(delay: float) -> None:
    with pytest.raises(ValueError, match="api_retry_sleep_secs"):
        RetryFunctionWrapper(Mock(), 3, delay)


def test_unknown_failure_is_not_retried_or_slept_after() -> None:
    error = ValueError("permanent")
    func = Mock(side_effect=error)
    with patch("time.sleep") as sleep, pytest.raises(ValueError) as caught:
        RetryFunctionWrapper(func, 10, 1)()
    assert caught.value is error
    func.assert_called_once()
    sleep.assert_not_called()


def test_predicate_can_explicitly_allow_a_failure_and_exhaust_without_final_sleep() -> (
    None
):
    error = ValueError("temporary in this caller")
    func = Mock(side_effect=error)
    with (
        patch("time.sleep") as sleep,
        pytest.raises(MaxAttemptsExceededError) as caught,
    ):
        RetryFunctionWrapper(func, 3, 0.5, retry_if=lambda _: True)()
    assert caught.value.__cause__ is error
    assert func.call_count == 3
    assert [call.args for call in sleep.call_args_list] == [(0.5,), (0.5,)]


def test_chain_prefers_explicit_cause_and_stops_at_cycles() -> None:
    outer, inner = RuntimeError(), ConnectionError()
    outer.__cause__ = inner
    outer.__context__ = ValueError()
    inner.__cause__ = outer
    assert list(exception_chain(outer)) == [outer, inner]
