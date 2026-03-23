# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith._internal.collections.sequenced_buffer import SequencedBuffer


def test_sequenced_buffer() -> None:
    buffer = SequencedBuffer[str]()

    # Test adding items at the next index
    buffer.add_at(0, "foo")
    assert len(buffer) == 1
    assert list(buffer.yield_in_order()) == ["foo"]
    assert len(buffer) == 0

    # Test adding items at subsequent indices
    buffer.add_at(2, "bar")
    assert len(buffer) == 1
    assert not list(buffer.yield_in_order())
    assert len(buffer) == 1

    buffer.add_at(4, "baz")
    assert len(buffer) == 2
    assert not list(buffer.yield_in_order())
    assert len(buffer) == 2

    # Fill in the gap at index 1.
    buffer.add_at(1, "qux")
    assert list(buffer.yield_in_order()) == ["qux", "bar"]
    assert len(buffer) == 1
    assert not list(buffer.yield_in_order())
    assert len(buffer) == 1

    # Fill in the gap at index 3.
    buffer.add_at(3, "foo")
    assert list(buffer.yield_in_order()) == ["foo", "baz"]
    assert len(buffer) == 0

    # Buffer should be empty now
    assert not list(buffer.yield_in_order())
    assert len(buffer) == 0


def test_sequenced_buffer_no_repeated_indices() -> None:
    buffer = SequencedBuffer[complex]()
    buffer.add_at(0, 1 + 2j)
    buffer.add_at(1, 3 + 4j)

    with pytest.raises(AssertionError, match="Item at index 1 already exists"):
        buffer.add_at(1, 5 + 6j)

    assert list(buffer.yield_in_order()) == [1 + 2j, 3 + 4j]
    assert len(buffer) == 0

    with pytest.raises(
        AssertionError, match="Adding item at prior index 0 is not allowed"
    ):
        buffer.add_at(0, 7 + 8j)
