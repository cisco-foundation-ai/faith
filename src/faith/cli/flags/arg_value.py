# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides types for wrapping parsed argument values that track their origin."""

from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

_T = TypeVar("_T")


class ArgValue(ABC, Generic[_T]):
    """A wrapper around a parsed argument value that tracks its origin."""

    def __init__(self, value: _T) -> None:
        self._value = value

    @property
    def value(self) -> _T:
        """Return the wrapped value."""
        return self._value

    @property
    @abstractmethod
    def is_default(self) -> bool:
        """Return True if this value was not explicitly set by the user."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._value!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ArgValue):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)


class DefaultValue(ArgValue[_T]):
    """An argument value that represents a default (not user-provided)."""

    @property
    def is_default(self) -> bool:
        return True


class UserValue(ArgValue[_T]):
    """An argument value explicitly provided by the user.

    Also serves as the argparse ``type=`` callable: instantiate with a
    parse function, then argparse calls the instance with the raw string
    which sets ``_value`` and returns ``self``.
    """

    @property
    def is_default(self) -> bool:
        return False


class UserValueType(Generic[_T]):
    """A type that can be parsed from a string and indicates it is user-provided."""

    def __init__(self, parse: Callable[[str], _T]) -> None:
        self._parse = parse

    def __call__(self, s: str) -> UserValue[_T]:
        """Parse the string `s` and return a UserValue wrapping the parsed value."""
        return UserValue(self._parse(s))


class TypeWithDefault(Generic[_T]):
    """A type that can be parsed from a string and has a default value."""

    def __init__(self, parse_fn: Callable[[str], _T], default_value: _T) -> None:
        """Initialize the TypeWithDefault with a parsing function and a default value."""
        self._parse_fn = parse_fn
        self._default_value = default_value

    def __call__(self, arg: str | None) -> _T:
        """Parse the `arg` or return the default value if `arg` is None."""
        if arg is None:
            return self._default_value
        return self._parse_fn(arg)
