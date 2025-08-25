# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides an abstract base class for transformations on iterators."""
from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

# Generic I/O TypeVars for a Transform.
In = TypeVar("In")
Out = TypeVar("Out")
Other = TypeVar("Other")


class Transform(ABC, Generic[In, Out]):
    """Base class for a callable that transforms an iterator of type I to type O."""

    def __rrshift__(self, src: Iterable[In]) -> Iterable[Out]:
        """Allow chaining of iterators with the >> operator."""
        return self(src)

    def __or__(self, other: "Transform[Out, Other]") -> "Transform[In, Other]":
        """Allow composition of transforms with the | operator."""
        return _TransformCompose(self, other)

    @abstractmethod
    def __call__(self, src: Iterable[In]) -> Iterable[Out]:
        """Protocol for a callable that transforms a iterator of type I to type O."""


class _TransformCompose(Transform[In, Other], Generic[In, Out, Other]):
    """A transform that composes two transforms into a new transform."""

    def __init__(self, first: Transform[In, Out], second: Transform[Out, Other]):
        """Initialize with two transforms."""
        self.first = first
        self.second = second

    def __call__(self, src: Iterable[In]) -> Iterable[Other]:
        """Apply the first transform and then the second transform."""
        return src >> self.first >> self.second


class IsoTransform(Transform[In, In], Generic[In]):
    """A transform with the same input and output types."""


class IdentityTransform(IsoTransform[In], Generic[In]):
    """A transform that returns items from an iterator as is."""

    def __rrshift__(self, src: Iterable[In]) -> Iterable[In]:
        """Allow chaining of iterators with the >> operator."""
        # Simplify chaining in the >> operator by excising identities.
        return src

    def __call__(self, src: Iterable[In]) -> Iterable[In]:
        """Identity transform that returns items from the `src` iterator as is."""
        yield from src


class Reducer(ABC, Generic[In, Out]):
    """Base class for a reducer that aggregates items from an iterator."""

    def __rrshift__(self, src: Iterable[In]) -> Out:
        """Allow piping of iterators to a reducer with the >> operator."""
        return self(src)

    @abstractmethod
    def __call__(self, src: Iterable[In]) -> Out:
        """Aggregate items from the `src` iterator and return a single output."""


class DevNullReducer(Reducer[In, None], Generic[In]):
    """A reducer that discards all items from the iterator.

    This is useful for cases where you want to process an iterator but do not
    care about the output, since iterators are lazy and do not iterate until
    they are consumed.
    """

    def __call__(self, src: Iterable[In]) -> None:
        """Discard all items from the `src` iterator."""
        for _ in src:
            pass
