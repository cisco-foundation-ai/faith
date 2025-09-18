# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides an abstract base class for transformations on iterators."""
from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

# Generic I/O TypeVars for a Transform.
_IN = TypeVar("_IN")
_OUT = TypeVar("_OUT")
_OTHER = TypeVar("_OTHER")


class Transform(ABC, Generic[_IN, _OUT]):
    """Base class for a callable that transforms an iterator of type I to type O."""

    def __rrshift__(self, src: Iterable[_IN]) -> Iterable[_OUT]:
        """Allow chaining of iterators with the >> operator."""
        return self(src)

    def __or__(self, other: "Transform[_OUT, _OTHER]") -> "Transform[_IN, _OTHER]":
        """Allow composition of transforms with the | operator."""
        return _TransformCompose(self, other)

    @abstractmethod
    def __call__(self, src: Iterable[_IN]) -> Iterable[_OUT]:
        """Protocol for a callable that transforms a iterator of type I to type O."""


class _TransformCompose(Transform[_IN, _OTHER], Generic[_IN, _OUT, _OTHER]):
    """A transform that composes two transforms into a new transform."""

    def __init__(self, first: Transform[_IN, _OUT], second: Transform[_OUT, _OTHER]):
        """Initialize with two transforms."""
        self.first = first
        self.second = second

    def __call__(self, src: Iterable[_IN]) -> Iterable[_OTHER]:
        """Apply the first transform and then the second transform."""
        return src >> self.first >> self.second


class IsoTransform(Transform[_IN, _IN], Generic[_IN]):
    """A transform with the same input and output types."""


class IdentityTransform(IsoTransform[_IN], Generic[_IN]):
    """A transform that returns items from an iterator as is."""

    def __rrshift__(self, src: Iterable[_IN]) -> Iterable[_IN]:
        """Allow chaining of iterators with the >> operator."""
        # Simplify chaining in the >> operator by excising identities.
        return src

    def __call__(self, src: Iterable[_IN]) -> Iterable[_IN]:
        """Identity transform that returns items from the `src` iterator as is."""
        yield from src


class Reducer(ABC, Generic[_IN, _OUT]):
    """Base class for a reducer that aggregates items from an iterator."""

    def __rrshift__(self, src: Iterable[_IN]) -> _OUT:
        """Allow piping of iterators to a reducer with the >> operator."""
        return self(src)

    @abstractmethod
    def __call__(self, src: Iterable[_IN]) -> _OUT:
        """Aggregate items from the `src` iterator and return a single output."""


class DevNullReducer(Reducer[_IN, None], Generic[_IN]):
    """A reducer that discards all items from the iterator.

    This is useful for cases where you want to process an iterator but do not
    care about the output, since iterators are lazy and do not iterate until
    they are consumed.
    """

    def __call__(self, src: Iterable[_IN]) -> None:
        """Discard all items from the `src` iterator."""
        for _ in src:
            pass
