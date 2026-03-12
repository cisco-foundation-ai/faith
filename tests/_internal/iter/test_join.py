# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from operator import itemgetter

import pytest
from pytest_unordered import unordered

from faith._internal.iter.join import (
    InnerJoinTransform,
    LeftJoinTransform,
    OuterJoinTransform,
    RightJoinTransform,
)

# Key extractor for (key, value) tuples used throughout these tests.
_key = itemgetter(0)


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ([], [], []),
        ([(1, "a")], [], []),
        ([], [(1, "x")], []),
        ([(1, "a"), (2, "b")], [(3, "x")], []),
        (
            [(1, "a"), (2, "b")],
            [(2, "x"), (3, "y")],
            [((2, "b"), (2, "x"))],
        ),
        (
            [(1, "a"), (1, "b")],
            [(1, "x"), (1, "y")],
            [
                ((1, "a"), (1, "x")),
                ((1, "a"), (1, "y")),
                ((1, "b"), (1, "x")),
                ((1, "b"), (1, "y")),
            ],
        ),
    ],
    ids=[
        "both-empty",
        "right-empty",
        "left-empty",
        "no-overlap",
        "partial",
        "cross-product",
    ],
)
def test_inner_join(left, right, expected) -> None:
    """Test that inner join yields only pairs where the key exists in both sides."""
    assert list(left >> InnerJoinTransform(right, on_key=_key)) == unordered(expected)


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ([], [], []),
        ([(1, "a")], [], [((1, "a"), None)]),
        ([], [(1, "x")], []),
        ([(1, "a"), (2, "b")], [(3, "x")], [((1, "a"), None), ((2, "b"), None)]),
        (
            [(1, "a"), (2, "b")],
            [(2, "x"), (3, "y")],
            [((1, "a"), None), ((2, "b"), (2, "x"))],
        ),
        (
            [(1, "a"), (1, "b")],
            [(1, "x"), (1, "y")],
            [
                ((1, "a"), (1, "x")),
                ((1, "a"), (1, "y")),
                ((1, "b"), (1, "x")),
                ((1, "b"), (1, "y")),
            ],
        ),
    ],
    ids=[
        "both-empty",
        "right-empty",
        "left-empty",
        "no-overlap",
        "partial",
        "cross-product",
    ],
)
def test_left_join(left, right, expected) -> None:
    """Test that left join preserves all left items, with None for unmatched rights."""
    assert list(left >> LeftJoinTransform(right, on_key=_key)) == unordered(expected)


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ([], [], []),
        ([(1, "a")], [], []),
        ([], [(1, "x")], [(None, (1, "x"))]),
        ([(1, "a"), (2, "b")], [(3, "x")], [(None, (3, "x"))]),
        (
            [(1, "a"), (2, "b")],
            [(2, "x"), (3, "y")],
            [((2, "b"), (2, "x")), (None, (3, "y"))],
        ),
        (
            [(1, "a"), (1, "b")],
            [(1, "x"), (1, "y")],
            [
                ((1, "a"), (1, "x")),
                ((1, "a"), (1, "y")),
                ((1, "b"), (1, "x")),
                ((1, "b"), (1, "y")),
            ],
        ),
    ],
    ids=[
        "both-empty",
        "right-empty",
        "left-empty",
        "no-overlap",
        "partial",
        "cross-product",
    ],
)
def test_right_join(left, right, expected) -> None:
    """Test that right join preserves all right items, with None for unmatched lefts."""
    assert list(left >> RightJoinTransform(right, on_key=_key)) == unordered(expected)


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ([], [], []),
        ([(1, "a")], [], [((1, "a"), None)]),
        ([], [(1, "x")], [(None, (1, "x"))]),
        (
            [(1, "a"), (2, "b")],
            [(3, "x")],
            [((1, "a"), None), ((2, "b"), None), (None, (3, "x"))],
        ),
        (
            [(1, "a"), (2, "b")],
            [(2, "x"), (3, "y")],
            [((1, "a"), None), ((2, "b"), (2, "x")), (None, (3, "y"))],
        ),
        (
            [(1, "a"), (1, "b")],
            [(1, "x"), (1, "y")],
            [
                ((1, "a"), (1, "x")),
                ((1, "a"), (1, "y")),
                ((1, "b"), (1, "x")),
                ((1, "b"), (1, "y")),
            ],
        ),
    ],
    ids=[
        "both-empty",
        "right-empty",
        "left-empty",
        "no-overlap",
        "partial",
        "cross-product",
    ],
)
def test_outer_join(left, right, expected) -> None:
    """Test that outer join preserves all items from both sides."""
    assert list(left >> OuterJoinTransform(right, on_key=_key)) == unordered(expected)
