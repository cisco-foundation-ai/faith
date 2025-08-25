# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Specialized collections not available in the standard library.

The `SequencedBuffer` class is a specialized collection for collecting items
out of order and returning them in sequence. It is useful for aligning outputs
created by a set of threads or processes that may not complete in the order
they were started.
"""
from heapq import heappop, heappush
from typing import Generic, TypeVar

T = TypeVar("T")


class SequencedBuffer(Generic[T]):
    """A buffer that collects items out of order and accesses them in sequence.

    This class allows adding items at specific indices and retrieving them in
    the order of their indices. It is useful for scenarios where items are
    produced asynchronously and may not arrive in the order they were requested.
    It is particularly useful for collecting and aligning outputs from threads.

    This class is not thread-safe, though.
    """

    def __init__(self) -> None:
        """Create an empty sequenced buffer."""
        self._queue: list[tuple[int, T]] = []
        self._next_index = 0

    def add_at(self, index: int, item: T) -> None:
        """Populate the buffer with an item for the given index."""
        assert (
            index >= self._next_index
        ), f"Adding item at prior index {index} is not allowed."
        assert index not in (
            i[0] for i in self._queue
        ), f"Item at index {index} already exists."
        heappush(self._queue, (index, item))

    def next_in_order(self) -> T | None:
        """Return the next item in order, or None if the next item is not available."""
        if not self._queue or self._queue[0][0] != self._next_index:
            return None
        _, item = heappop(self._queue)
        self._next_index += 1
        return item

    def __len__(self) -> int:
        """Return the number of items remaining in the buffer."""
        return len(self._queue)
