# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from faith._internal.records.reconciliation import (
    ReplacementStrategy,
    reconcile_records,
)
from faith._internal.records.types import Record, RecordStatus

CLEAN = RecordStatus.CLEAN
DIRTY = RecordStatus.DIRTY


def _rec(sample_hash: str, data_hash: str, **extra: Any) -> Record:
    """Build a minimal record for testing."""
    return {
        "metadata": {"data_hash": data_hash},
        "data": {"benchmark_sample_hash": sample_hash, **extra},
    }


@pytest.mark.parametrize(
    "existing, new, expected",
    [
        ([], [], []),
        ([], [_rec("a", "h1")], [(DIRTY, _rec("a", "h1"))]),
        ([_rec("a", "h1")], [], []),
        (
            [_rec("a", "h1", v="old")],
            [_rec("a", "h1", v="new")],
            [(CLEAN, _rec("a", "h1", v="old"))],
        ),
        (
            [_rec("a", "h_old", v="old")],
            [_rec("a", "h_new", v="new")],
            [(CLEAN, _rec("a", "h_old", v="old"))],
        ),
        (
            [_rec("a", "h1")],
            [_rec("a", "h1"), _rec("b", "h2")],
            [(CLEAN, _rec("a", "h1")), (DIRTY, _rec("b", "h2"))],
        ),
        (
            [_rec("c", "h3"), _rec("a", "h1")],
            [_rec("a", "h1"), _rec("b", "h2"), _rec("c", "h3")],
            [
                (CLEAN, _rec("a", "h1")),
                (DIRTY, _rec("b", "h2")),
                (CLEAN, _rec("c", "h3")),
            ],
        ),
    ],
    ids=[
        "both-empty",
        "empty-old-stream",
        "empty-new-stream",
        "match-same-hash",
        "match-different-hash",
        "mixed-matched-and-unmatched",
        "preserves-input-order",
    ],
)
def test_strategy_never(
    existing: list[Record],
    new: list[Record],
    expected: list[tuple[RecordStatus, Record]],
) -> None:
    """NEVER: always keep the existing record when one matches."""
    assert (
        list(new >> reconcile_records(existing, ReplacementStrategy.NEVER)) == expected
    )


@pytest.mark.parametrize(
    "existing, new, expected",
    [
        ([], [], []),
        ([], [_rec("a", "h1")], [(DIRTY, _rec("a", "h1"))]),
        ([_rec("a", "h1")], [], []),
        (
            [_rec("a", "h1", v="old")],
            [_rec("a", "h1", v="new")],
            [(DIRTY, _rec("a", "h1", v="new"))],
        ),
        (
            [_rec("a", "h_old", v="old")],
            [_rec("a", "h_new", v="new")],
            [(DIRTY, _rec("a", "h_new", v="new"))],
        ),
        (
            [_rec("a", "h1", v="old")],
            [_rec("a", "h1", v="new"), _rec("b", "h2")],
            [(DIRTY, _rec("a", "h1", v="new")), (DIRTY, _rec("b", "h2"))],
        ),
    ],
    ids=[
        "both-empty",
        "empty-old-stream",
        "empty-new-stream",
        "match-same-hash",
        "match-different-hash",
        "mixed-all-dirty",
    ],
)
def test_strategy_always(
    existing: list[Record],
    new: list[Record],
    expected: list[tuple[RecordStatus, Record]],
) -> None:
    """ALWAYS: always take the new record, regardless of match."""
    assert (
        list(new >> reconcile_records(existing, ReplacementStrategy.ALWAYS)) == expected
    )


@pytest.mark.parametrize(
    "existing, new, expected",
    [
        ([], [], []),
        ([], [_rec("a", "h1")], [(DIRTY, _rec("a", "h1"))]),
        ([_rec("a", "h1")], [], []),
        (
            [_rec("a", "h1", v="old")],
            [_rec("a", "h1", v="new")],
            [(CLEAN, _rec("a", "h1", v="old"))],
        ),
        (
            [_rec("a", "h_old", v="old")],
            [_rec("a", "h_new", v="new")],
            [(DIRTY, _rec("a", "h_new", v="new"))],
        ),
        (
            [_rec("a", "h1"), _rec("b", "h_old")],
            [_rec("a", "h1"), _rec("b", "h_new"), _rec("c", "h3")],
            [
                (CLEAN, _rec("a", "h1")),
                (DIRTY, _rec("b", "h_new")),
                (DIRTY, _rec("c", "h3")),
            ],
        ),
    ],
    ids=[
        "both-empty",
        "empty-old-stream",
        "empty-new-stream",
        "match-same-hash",
        "match-different-hash",
        "mixed-clean-dirty-and-unmatched",
    ],
)
def test_strategy_if_data_hash_differs(
    existing: list[Record],
    new: list[Record],
    expected: list[tuple[RecordStatus, Record]],
) -> None:
    """IF_DATA_HASH_DIFFERS: take new only when the data_hash changed."""
    assert (
        list(
            new >> reconcile_records(existing, ReplacementStrategy.IF_DATA_HASH_DIFFERS)
        )
        == expected
    )
