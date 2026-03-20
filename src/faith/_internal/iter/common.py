# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Common transforms over iterators."""

from typing import Callable, Generic, TypeVar

from faith._internal.iter.transform import Mapping

_IN = TypeVar("_IN")
_OUT = TypeVar("_OUT")


class GetAttrTransform(Mapping[_IN, _OUT], Generic[_IN, _OUT]):
    """A transform that gets a specified attribute from each element in the iterator."""

    def __init__(self, attr_name: str) -> None:
        """Initialize with the name of the attribute to get."""
        super().__init__()
        self._attr_name = attr_name

    def _map_fn(self, element: _IN) -> _OUT:
        """Get the specified attribute from the element."""
        return getattr(element, self._attr_name)


class Functor(Mapping[_IN, _OUT], Generic[_IN, _OUT]):
    """A transform that applies a specified function to each element in the iterator."""

    def __init__(self, func: Callable[[_IN], _OUT]) -> None:
        """Initialize with the function to apply."""
        super().__init__()
        self._func = func

    def _map_fn(self, element: _IN) -> _OUT:
        """Apply the function to the element."""
        return self._func(element)
