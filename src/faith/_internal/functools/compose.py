# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""A module for composing functions.

This module provides functionality to compose multiple functions into a single callable,
implemented using a class-based approach to make function composition pickleable;
this is particularly useful in multiprocessing contexts.
"""

from functools import reduce
from typing import Any, Callable, Generic, TypeVar

R1 = TypeVar("R1")
R2 = TypeVar("R2")


class _FunctionComposition(Generic[R1, R2]):
    """A class for composing two functions into a single callable."""

    def __init__(self, f: Callable[[R1], R2], g: Callable[..., R1]) -> None:
        """Initialize with two functions to compose."""
        self._f = f
        self._g = g

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the composed functions in sequence."""
        return self._f(self._g(*args, **kwargs))


def compose(*funcs: Callable[..., Any]) -> Callable[..., Any]:
    """Compose multiple functions into a single callable.

    The functions are applied from right to left; i.e., the last function is applied
    first to the input arguments, and each preceding function is applied to the
    subsequent result.

    Args:
        *funcs: A variable number of functions to compose.

    Returns:
        A single callable that represents the composition of the input functions.

    Raises:
        ValueError: If no functions are provided for composition.
    """
    if not funcs:
        raise ValueError("At least one function must be provided for composition.")
    return reduce(_FunctionComposition, funcs)
