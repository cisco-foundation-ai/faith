# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Iterator
from queue import Empty, Queue
from threading import Semaphore
from types import TracebackType
from typing import Generic, TypeVar

_T = TypeVar("_T")

_SENTINEL = object()


class QueueIterable(Iterable[_T], Generic[_T]):
    """An iterable backed by an internal queue.

    Items are pushed via ``put`` and iteration terminates after all expected
    ``close`` calls have been made.  Use as a context manager to ensure
    ``close`` is called on error.

    Args:
        num_producers: Number of ``close`` calls expected before iteration
            terminates.  Defaults to 1 for single-producer use.
    """

    def __init__(self, num_producers: int = 1) -> None:
        assert num_producers > 0, "num_producers must be positive"
        self._queue: Queue[_T | object] = Queue()
        self._sentinels: Queue[object] = Queue()
        self._close_semaphore = Semaphore(num_producers - 1)
        for _ in range(num_producers):
            self._sentinels.put(_SENTINEL)

    def put(self, item: _T) -> None:
        """Enqueue an item to be yielded by the iterator."""
        if self._sentinels.empty():
            raise RuntimeError("Cannot put items after close has been called.")
        self._queue.put(item)

    def _close(self) -> None:
        """Signal that one producer is done.

        Iteration terminates only after all expected producers have closed.
        """
        try:
            sentinel = self._sentinels.get_nowait()
        except Empty:
            raise RuntimeError("close() called more times than expected.") from None
        self._queue.put(sentinel)

    def __enter__(self) -> "QueueIterable[_T]":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._close()

    def __iter__(self) -> Iterator[_T]:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                # pylint: disable-next=consider-using-with
                if not self._close_semaphore.acquire(blocking=False):
                    break
                continue
            yield item  # type: ignore[misc]
