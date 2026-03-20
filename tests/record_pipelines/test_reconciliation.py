# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from faith._types.model.generation import GenerationMode
from faith._types.record.sample_record import (
    RecordStatus,
    ReplacementStrategy,
    SampleRecord,
)
from faith.record_pipelines.reconciliation import record_reconciler
from tests.benchmark.categories.fake_record_maker import make_fake_record

FRESH = RecordStatus.FRESH
STALE = RecordStatus.STALE


def _rec(
    sample_hash: str,
    data_hash: str,
    *,
    question: str = "",
    model_stats: dict | None = None,
) -> SampleRecord:
    """Build a minimal record for testing."""
    return make_fake_record(
        metadata={"data_hash": data_hash, "version": "v0.0.7"},
        data={"benchmark_sample_hash": sample_hash, "question": question},
        model_data={"prompt": "The quick brown", "answer_symbol_ids": {}}
        | (model_stats or {}),
    )


@pytest.mark.parametrize(
    "existing, new, expected",
    [
        ([], [], []),
        ([], [_rec("a", "h1")], [(STALE, _rec("a", "h1"))]),
        ([_rec("a", "h1")], [], []),
        (
            [
                _rec(
                    "a",
                    "h1",
                    question="old",
                    model_stats={"chat_comp": {"output_text": "foo"}},
                )
            ],
            [_rec("a", "h1", question="new")],
            [
                (
                    FRESH,
                    _rec(
                        "a",
                        "h1",
                        question="old",
                        model_stats={"chat_comp": {"output_text": "foo"}},
                    ),
                )
            ],
        ),
        (
            [_rec("a", "h_old", question="old")],
            [_rec("a", "h_new", question="new")],
            [(STALE, _rec("a", "h_old", question="old"))],
        ),
        (
            [_rec("a", "h1", model_stats={"next_token": {"output_text": "foo"}})],
            [_rec("a", "h1"), _rec("b", "h2")],
            [(STALE, _rec("a", "h1")), (STALE, _rec("b", "h2"))],
        ),
        (
            [
                _rec("c", "h3"),
                _rec("a", "h1", model_stats={"chat_comp": {"output_text": "foo"}}),
            ],
            [
                _rec("a", "h1"),
                _rec("b", "h2", model_stats={"next_token": {"output_text": "bar"}}),
                _rec("c", "h3"),
            ],
            [
                (
                    FRESH,
                    _rec("a", "h1", model_stats={"chat_comp": {"output_text": "foo"}}),
                ),
                (STALE, _rec("b", "h2")),
                (STALE, _rec("c", "h3")),
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
def test_record_reconciler_strategy_never(
    existing: list[SampleRecord],
    new: list[SampleRecord],
    expected: list[tuple[RecordStatus, SampleRecord]],
) -> None:
    """NEVER: always keep the existing record when one matches."""
    assert (
        list(
            new
            >> record_reconciler(
                existing, ReplacementStrategy.NEVER, GenerationMode.CHAT_COMP
            )
        )
        == expected
    )


@pytest.mark.parametrize(
    "existing, new, expected",
    [
        ([], [], []),
        ([], [_rec("a", "h1")], [(STALE, _rec("a", "h1"))]),
        ([_rec("a", "h1")], [], []),
        (
            [_rec("a", "h1", question="old")],
            [_rec("a", "h1", question="new")],
            [(STALE, _rec("a", "h1", question="new"))],
        ),
        (
            [_rec("a", "h_old", question="old")],
            [_rec("a", "h_new", question="new")],
            [(STALE, _rec("a", "h_new", question="new"))],
        ),
        (
            [_rec("a", "h1", question="old")],
            [_rec("a", "h1", question="new"), _rec("b", "h2")],
            [(STALE, _rec("a", "h1", question="new")), (STALE, _rec("b", "h2"))],
        ),
    ],
    ids=[
        "both-empty",
        "empty-old-stream",
        "empty-new-stream",
        "match-same-hash",
        "match-different-hash",
        "mixed-all-stale",
    ],
)
def test_record_reconciler_strategy_always(
    existing: list[SampleRecord],
    new: list[SampleRecord],
    expected: list[tuple[RecordStatus, SampleRecord]],
) -> None:
    """ALWAYS: always take the new record, regardless of match."""
    assert (
        list(
            new
            >> record_reconciler(
                existing, ReplacementStrategy.ALWAYS, GenerationMode.NEXT_TOKEN
            )
        )
        == expected
    )


@pytest.mark.parametrize(
    "existing, new, expected",
    [
        ([], [], []),
        ([], [_rec("a", "h1")], [(STALE, _rec("a", "h1"))]),
        ([_rec("a", "h1")], [], []),
        (
            [
                _rec(
                    "a",
                    "h1",
                    question="old",
                    model_stats={
                        "logits": [[{"token": "a", "token_id": 0, "logprob": -0.01}]]
                    },
                )
            ],
            [_rec("a", "h1", question="new")],
            [
                (
                    FRESH,
                    _rec(
                        "a",
                        "h1",
                        question="old",
                        model_stats={
                            "logits": [
                                [{"token": "a", "token_id": 0, "logprob": -0.01}]
                            ]
                        },
                    ),
                )
            ],
        ),
        (
            [_rec("a", "h_old", question="old")],
            [_rec("a", "h_new", question="new")],
            [(STALE, _rec("a", "h_new", question="new"))],
        ),
        (
            [_rec("a", "h1"), _rec("b", "h_old")],
            [
                _rec(
                    "a",
                    "h1",
                    model_stats={
                        "logits": [[{"token": "a", "token_id": 0, "logprob": -0.01}]]
                    },
                ),
                _rec("b", "h_new"),
                _rec("c", "h3"),
            ],
            [
                (STALE, _rec("a", "h1")),
                (STALE, _rec("b", "h_new")),
                (STALE, _rec("c", "h3")),
            ],
        ),
    ],
    ids=[
        "both-empty",
        "empty-old-stream",
        "empty-new-stream",
        "match-same-hash",
        "match-different-hash",
        "mixed-fresh-stale-and-unmatched",
    ],
)
def test_record_reconciler_strategy_if_hash_differs(
    existing: list[SampleRecord],
    new: list[SampleRecord],
    expected: list[tuple[RecordStatus, SampleRecord]],
) -> None:
    """IF_HASH_DIFFERS: take new only when the hash changed."""
    assert (
        list(
            new
            >> record_reconciler(
                existing,
                ReplacementStrategy.IF_HASH_DIFFERS,
                GenerationMode.LOGITS,
            )
        )
        == expected
    )
