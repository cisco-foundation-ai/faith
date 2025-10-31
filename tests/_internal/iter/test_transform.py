# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable

from faith._internal.iter.transform import (
    DevNullReducer,
    IdentityTransform,
    IsoTransform,
    Reducer,
    Transform,
)


class SquareTransform(IsoTransform[int]):
    """A simple transform that squares each integer in the generator."""

    def __call__(self, it: Iterable[int]) -> Iterable[int]:
        """Yield the square of each integer from the input generator."""
        for item in it:
            yield item * item


class IncrementTransform(IsoTransform[int]):
    """A simple transform that increments each integer in the generator."""

    def __call__(self, it: Iterable[int]) -> Iterable[int]:
        """Yield each integer incremented by one from the input generator."""
        for item in it:
            yield item + 1


class ToStringTransform(Transform[int, str]):
    """A transform that converts each integer to a string."""

    def __call__(self, it: Iterable[int]) -> Iterable[str]:
        """Yield the string representation of each integer from the input generator."""
        for item in it:
            yield str(item)


class SumReducer(Reducer[int, int]):
    """A reducer that sums all integers from the iterator."""

    def __call__(self, it: Iterable[int]) -> int:
        """Return the sum of all integers from the input generator."""
        return sum(item for item in it)


class StringConcatReducer(Reducer[str, str]):
    """A reducer that concatenates all strings from the iterator."""

    def __call__(self, it: Iterable[str]) -> str:
        """Return the concatenation of all strings from the input generator."""
        return "".join(item for item in it)


def test_transform_chaining() -> None:
    """Test chaining of transforms."""
    assert list(range(5) >> SquareTransform()) == [0, 1, 4, 9, 16]
    assert list(range(5) >> IncrementTransform()) == [1, 2, 3, 4, 5]
    assert list(range(5) >> SquareTransform() >> IncrementTransform()) == [
        1,
        2,
        5,
        10,
        17,
    ]
    assert list(range(5) >> IncrementTransform() >> SquareTransform()) == [
        1,
        4,
        9,
        16,
        25,
    ]
    assert list(range(5) >> SquareTransform() >> SquareTransform()) == [
        0,
        1,
        16,
        81,
        256,
    ]
    assert list(
        range(5) >> SquareTransform() >> IncrementTransform() >> ToStringTransform()
    ) == [
        "1",
        "2",
        "5",
        "10",
        "17",
    ]


def test_transform_chain_assignment() -> None:
    """Test that assignment operator works with chaining transforms."""
    data: Iterable[int] = range(5)
    data >>= SquareTransform()
    assert list(data) == [0, 1, 4, 9, 16]


def test_transform_composition() -> None:
    """Test composition of transforms."""
    sq_inc = SquareTransform() | IncrementTransform()
    assert list(range(5) >> sq_inc) == [1, 2, 5, 10, 17]
    inc_sq = IncrementTransform() | SquareTransform()
    assert list(range(5) >> inc_sq) == [1, 4, 9, 16, 25]
    sq_sq = SquareTransform() | SquareTransform()
    assert list(range(5) >> sq_sq) == [0, 1, 16, 81, 256]
    sq_inc_tostr = SquareTransform() | IncrementTransform() | ToStringTransform()
    assert list(range(5) >> sq_inc_tostr) == ["1", "2", "5", "10", "17"]


def test_transform_composition_assignment() -> None:
    """Test that assignment operator works with composing transforms."""
    tr: Transform[int, int] = SquareTransform()
    tr |= IncrementTransform()
    tr |= SquareTransform()
    assert list(range(5) >> tr) == [1, 4, 25, 100, 289]


def test_identity_transform() -> None:
    transform = IdentityTransform[str]()
    outputs = list(transform(s for s in ["foo", "bar"]))

    assert outputs == ["foo", "bar"]


def test_rshift_identity_transform() -> None:
    outputs = list((s for s in ["foo", "bar"]) >> IdentityTransform[str]())

    assert outputs == ["foo", "bar"]


def test_transform_chaining_reduction() -> None:
    assert range(5) >> SumReducer() == 10
    assert range(5) >> SquareTransform() >> SumReducer() == 30
    assert range(5) >> IncrementTransform() >> SumReducer() == 15
    assert range(5) >> SquareTransform() >> IncrementTransform() >> SumReducer() == 35
    assert range(5) >> IncrementTransform() >> SquareTransform() >> SumReducer() == 55
    assert range(5) >> SquareTransform() >> SquareTransform() >> SumReducer() == 354

    assert range(0) >> ToStringTransform() >> StringConcatReducer() == ""
    assert range(5) >> ToStringTransform() >> StringConcatReducer() == "01234"


def test_dev_null_reducer() -> None:
    """Test the DevNullReducer to ensure the iterator executes."""
    lst = [1, 2, 3, 4, 5]
    _ = (lst.pop(0) for _ in range(len(lst))) >> DevNullReducer[int]()
    assert len(lst) == 0, "List should be empty after using DevNullReducer."
