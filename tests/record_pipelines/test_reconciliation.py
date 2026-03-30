# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from faith._types.model.generation import GenerationMode
from faith._types.record.sample import RecordStatus, ReplacementStrategy, SampleRecord
from faith.record_pipelines.reconciliation import record_reconciler
from tests.benchmark.categories.fake_record_maker import make_fake_record

FRESH = RecordStatus.FRESH
STALE = RecordStatus.STALE


def _rec(
    sample_index: int,
    sample_hash: str,
    data_hash: str,
    *,
    question: str = "",
    model_stats: dict | None = None,
) -> SampleRecord:
    """Build a minimal record for testing."""
    return make_fake_record(
        metadata={"data_hash": data_hash, "version": "v0.0.7"},
        data={
            "benchmark_sample_index": sample_index,
            "benchmark_sample_hash": sample_hash,
            "question": question,
        },
        model_data={"prompt": "The quick brown", "answer_symbol_ids": {}}
        | (model_stats or {}),
    )


@pytest.mark.parametrize(
    "existing, new, expected",
    [
        ([], [], []),
        ([], [_rec(0, "a", "h1")], [(STALE, _rec(0, "a", "h1"))]),
        ([_rec(0, "a", "h1")], [], []),
        (
            [
                _rec(
                    0,
                    "a",
                    "h1",
                    question="old",
                    model_stats={"chat_comp": {"output_text": "foo"}},
                )
            ],
            [_rec(0, "a", "h1", question="new")],
            [
                (
                    FRESH,
                    _rec(
                        0,
                        "a",
                        "h1",
                        question="old",
                        model_stats={"chat_comp": {"output_text": "foo"}},
                    ),
                )
            ],
        ),
        (
            [_rec(1, "a", "h_old", question="old")],
            [_rec(1, "a", "h_new", question="new")],
            [(STALE, _rec(1, "a", "h_old", question="old"))],
        ),
        (
            [_rec(1, "a", "h1", model_stats={"next_token": {"output_text": "foo"}})],
            [_rec(1, "a", "h1"), _rec(0, "b", "h2")],
            [(STALE, _rec(1, "a", "h1")), (STALE, _rec(0, "b", "h2"))],
        ),
        (
            [
                _rec(2, "c", "h3"),
                _rec(0, "a", "h1", model_stats={"chat_comp": {"output_text": "foo"}}),
            ],
            [
                _rec(0, "a", "h1"),
                _rec(1, "b", "h2", model_stats={"next_token": {"output_text": "bar"}}),
                _rec(2, "c", "h3"),
            ],
            [
                (
                    FRESH,
                    _rec(
                        0, "a", "h1", model_stats={"chat_comp": {"output_text": "foo"}}
                    ),
                ),
                (STALE, _rec(1, "b", "h2")),
                (STALE, _rec(2, "c", "h3")),
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
        ([], [_rec(0, "a", "h1")], [(STALE, _rec(0, "a", "h1"))]),
        ([_rec(2, "a", "h1")], [], []),
        (
            [_rec(1, "a", "h1", question="old")],
            [_rec(1, "a", "h1", question="new")],
            [(STALE, _rec(1, "a", "h1", question="new"))],
        ),
        (
            [_rec(0, "a", "h_old", question="old")],
            [_rec(0, "a", "h_new", question="new")],
            [(STALE, _rec(0, "a", "h_new", question="new"))],
        ),
        (
            [_rec(1, "a", "h1", question="old")],
            [_rec(1, "a", "h1", question="new"), _rec(0, "b", "h2")],
            [(STALE, _rec(1, "a", "h1", question="new")), (STALE, _rec(0, "b", "h2"))],
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
        ([], [_rec(0, "a", "h1")], [(STALE, _rec(0, "a", "h1"))]),
        ([_rec(1, "a", "h1")], [], []),
        (
            [
                _rec(
                    1,
                    "a",
                    "h1",
                    question="old",
                    model_stats={
                        "logits": [[{"token": "a", "token_id": 0, "logprob": -0.01}]]
                    },
                )
            ],
            [_rec(1, "a", "h1", question="new")],
            [
                (
                    FRESH,
                    _rec(
                        1,
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
            [_rec(0, "a", "h_old", question="old")],
            [_rec(0, "a", "h_new", question="new")],
            [(STALE, _rec(0, "a", "h_new", question="new"))],
        ),
        (
            [_rec(1, "a", "h1"), _rec(0, "b", "h_old")],
            [
                _rec(
                    1,
                    "a",
                    "h1",
                    model_stats={
                        "logits": [[{"token": "a", "token_id": 0, "logprob": -0.01}]]
                    },
                ),
                _rec(0, "b", "h_new"),
                _rec(2, "c", "h3"),
            ],
            [
                (STALE, _rec(1, "a", "h1")),
                (STALE, _rec(0, "b", "h_new")),
                (STALE, _rec(2, "c", "h3")),
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


def test_duplicate_sample_hash_no_cross_product() -> None:
    """Regression: https://github.com/RobustIntelligence/faith/issues/476

    When benchmark data has rows with identical column values, multiple records share
    the same benchmark_sample_hash but have different benchmark_sample_index values.
    The reconciler must match 1-to-1 by (index, hash), to prevent an N*M
    cross-product that causes exponential growth of STALE records on each re-run.
    """
    chat_resp = {"chat_comp": {"output_text": "response"}}

    # Two existing records share sample_hash "H" but have different indices.
    existing = [
        _rec(5, "H", "d5", model_stats=chat_resp),
        _rec(10, "H", "d10", model_stats=chat_resp),
    ]

    assert list(
        [
            _rec(5, "H", "d5"),
            _rec(10, "H", "d10"),
        ]
        >> record_reconciler(
            existing, ReplacementStrategy.IF_HASH_DIFFERS, GenerationMode.CHAT_COMP
        )
    ) == [
        (FRESH, _rec(5, "H", "d5", model_stats=chat_resp)),
        (FRESH, _rec(10, "H", "d10", model_stats=chat_resp)),
    ]


def test_duplicate_sample_hash_stable_across_runs() -> None:
    """Regression Test: https://github.com/RobustIntelligence/faith/issues/476

    Check that repeated reconciliation with duplicate hashes does not grow the records.
    """
    chat_resp = {"chat_comp": {"output_text": "response"}}
    mode = GenerationMode.CHAT_COMP
    strategy = ReplacementStrategy.IF_HASH_DIFFERS

    # Run 1: no existing records — everything is STALE.
    assert list(
        [
            _rec(0, "H", "d0"),
            _rec(1, "H", "d1"),
            _rec(2, "U", "d2"),
        ]
        >> record_reconciler([], strategy, mode)
    ) == [
        (STALE, _rec(0, "H", "d0")),
        (STALE, _rec(1, "H", "d1")),
        (STALE, _rec(2, "U", "d2")),
    ]

    # Simulate re-running after model responses have been logged — all should be FRESH.
    run2 = list(
        [
            _rec(0, "H", "d0"),
            _rec(1, "H", "d1"),
            _rec(2, "U", "d2"),
        ]
        >> record_reconciler(
            [
                _rec(0, "H", "d0", model_stats=chat_resp),
                _rec(1, "H", "d1", model_stats=chat_resp),
                _rec(2, "U", "d2", model_stats=chat_resp),
            ],
            strategy,
            mode,
        )
    )
    assert run2 == [
        (FRESH, _rec(0, "H", "d0", model_stats=chat_resp)),
        (FRESH, _rec(1, "H", "d1", model_stats=chat_resp)),
        (FRESH, _rec(2, "U", "d2", model_stats=chat_resp)),
    ]

    # Run 3: feed run2 output back — count must stay at 3, all FRESH.
    assert list(
        [
            _rec(0, "H", "d0"),
            _rec(1, "H", "d1"),
            _rec(2, "U", "d2"),
        ]
        >> record_reconciler([rec for _, rec in run2], strategy, mode)
    ) == [
        (FRESH, _rec(0, "H", "d0", model_stats=chat_resp)),
        (FRESH, _rec(1, "H", "d1", model_stats=chat_resp)),
        (FRESH, _rec(2, "U", "d2", model_stats=chat_resp)),
    ]
