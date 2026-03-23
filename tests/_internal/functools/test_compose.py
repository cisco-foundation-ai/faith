# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the compose function."""

import pytest

from faith._internal.functools.compose import compose


def _add_one(x: int) -> int:
    return x + 1


def _multiply_by_two(x: int) -> int:
    return x * 2


def _square(x: int) -> int:
    return x * x


def _to_string(x: int) -> str:
    return str(x)


def _string_length(s: str) -> int:
    return len(s)


def _add_two_numbers(x: int, y: int) -> int:
    return x + y


def _negate(x: int) -> int:
    return -x


@pytest.mark.parametrize(
    "funcs,args,expected",
    [
        ([_add_one], (5,), 6),
        ([_multiply_by_two, _add_one], (5,), 12),
        ([_add_one, _multiply_by_two], (5,), 11),
        ([_square, _multiply_by_two, _add_one], (3,), 64),
        ([_add_one, _square, _multiply_by_two], (3,), 37),
        ([_add_one, _multiply_by_two, _square, _add_one], (2,), 19),
        ([_negate, _negate], (5,), 5),
        ([_to_string, _string_length, _to_string], (12345,), "5"),
        ([_multiply_by_two, _add_two_numbers], (3, 4), 14),
        ([_add_one, _add_two_numbers], (5, 10), 16),
        ([_add_one] * 2, (0,), 2),
        ([_add_one] * 5, (0,), 5),
        ([_add_one] * 100, (0,), 100),
    ],
)
def test_compose(funcs, args, expected):
    """Test function composition with various combinations."""
    composed = compose(*funcs)
    assert composed(*args) == expected


def test_compose_no_functions():
    """Test composing with no functions is an error."""
    with pytest.raises(
        ValueError, match="At least one function must be provided for composition."
    ):
        compose()
