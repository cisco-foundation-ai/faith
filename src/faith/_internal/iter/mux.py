# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from enum import Enum
from typing import Generic, Iterable, TypeVar

from faith._internal.iter.transform import Transform

# Generic I/O TypeVars for the MuxTransform.
Kind = TypeVar("Kind", bound=Enum)
In = TypeVar("In")
Out = TypeVar("Out")


class MuxTransform(Transform[tuple[Kind, In], Out], Generic[Kind, In, Out]):
    """A transform that routes input items to different transforms based on their kind."""

    def __init__(self, transform_map: dict[Kind, Transform[In, Out]]):
        """Initialize with a mapping of kinds to transforms."""
        self._transform_map = transform_map

    def __call__(self, src: Iterable[tuple[Kind, In]]) -> Iterable[Out]:
        # Route each item to its corresponding transform based on the key function.
        # Create a dictionary to hold iterators for each key.
        order_indices = []
        iterators = defaultdict(list)
        for kind, item in src:
            iterators[kind].append(item)
            order_indices.append(kind)
        assert set(iterators.keys()).issubset(
            self._transform_map.keys()
        ), f"All kinds must have a corresponding transform defined in the transform map; missing transforms for: {set(iterators.keys()) - set(self._transform_map.keys())}"

        # Apply the corresponding transform to each key's items.
        transformed_items = {
            key: iter(it >> self._transform_map[key]) for key, it in iterators.items()
        }
        # Merge the transformed items and yield them in the order of their keys.
        for key in order_indices:
            if key in transformed_items:
                yield next(transformed_items[key])

        for it in transformed_items.values():
            # Assert that the iterator is exhausted.
            try:
                next(it)
            except StopIteration:
                continue
            else:
                raise RuntimeError("Iterator was not exhausted after merging.")
