# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from threading import Thread

import pytest

from faith._internal.collections.queue_iterable import QueueIterable


def test_queue_iterable_basic_iteration() -> None:
    """Items put before close are yielded in order."""
    qi: QueueIterable[int] = QueueIterable()
    with qi:
        for i in range(5):
            qi.put(i)
    assert list(qi) == [0, 1, 2, 3, 4]


def test_queue_iterable_empty() -> None:
    """An immediate close yields no items."""
    qi: QueueIterable[int] = QueueIterable()
    with qi:
        pass
    assert not list(qi)


def test_queue_iterable_blocks_until_items_available() -> None:
    """The iterable blocks on an empty queue until items arrive from another thread."""
    qi: QueueIterable[int] = QueueIterable()
    results: list[int] = []

    def producer() -> None:
        with qi:
            for i in range(3):
                qi.put(i)

    t = Thread(target=producer)
    t.start()
    for item in qi:
        results.append(item)
    t.join()

    assert results == [0, 1, 2]


def test_queue_iterable_put_after_close_raises() -> None:
    """Putting an item after close raises RuntimeError."""
    qi: QueueIterable[int] = QueueIterable()
    with qi:
        pass
    with pytest.raises(RuntimeError, match="Cannot put items after close"):
        qi.put(1)


def test_queue_iterable_extra_close_raises() -> None:
    """Calling close more times than expected raises RuntimeError."""
    qi: QueueIterable[str] = QueueIterable()
    qi.close()
    with pytest.raises(
        RuntimeError, match="close\\(\\) called more times than expected"
    ):
        qi.close()


def test_queue_iterable_context_manager_closes_on_error() -> None:
    """The iterable is closed even if the producer raises inside the context."""
    qi: QueueIterable[int] = QueueIterable()
    try:
        with qi:
            qi.put(1)
            raise ValueError("boom")
    except ValueError:
        pass
    assert list(qi) == [1]


def test_queue_iterable_multiple_producers() -> None:
    """Multiple producer threads each close independently."""
    qi: QueueIterable[int] = QueueIterable(num_producers=3)
    results: list[int] = []

    def producer(start: int) -> None:
        with qi:
            for i in range(start, start + 3):
                qi.put(i)

    threads = [Thread(target=producer, args=(i * 3,)) for i in range(3)]
    for t in threads:
        t.start()
    for item in qi:
        results.append(item)
    for t in threads:
        t.join()

    assert sorted(results) == list(range(9))
