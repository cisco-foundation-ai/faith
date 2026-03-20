# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0


from faith._internal.iter.common import Functor
from faith._types.config.patterns import AnswerFormat
from faith._types.record.sample_record import RecordStatus, ReplacementStrategy
from faith._types.record.stats import StatsRecord
from faith.record_pipelines.screening import stats_screener
from tests.benchmark.categories.fake_record_maker import make_fake_record

_FAKE_STATS = StatsRecord(label="A", prediction="A", answer_format=AnswerFormat.PROPER)


def test_stats_screener_never_and_always() -> None:
    # Test with replacement strategy NEVER.
    assert not list([] >> stats_screener(ReplacementStrategy.NEVER))
    assert list(
        [make_fake_record(stats=_FAKE_STATS), make_fake_record(stats=None)]
        >> stats_screener(ReplacementStrategy.NEVER)
        >> Functor[tuple[RecordStatus, StatsRecord], RecordStatus](lambda x: x[0])
    ) == [
        RecordStatus.FRESH,
        RecordStatus.STALE,
    ]

    # Test with replacement strategy ALWAYS.
    assert not list([] >> stats_screener(ReplacementStrategy.ALWAYS))
    assert list(
        [make_fake_record(stats=_FAKE_STATS), make_fake_record(stats=None)]
        >> stats_screener(ReplacementStrategy.ALWAYS)
        >> Functor[tuple[RecordStatus, StatsRecord], RecordStatus](lambda x: x[0])
    ) == [
        RecordStatus.STALE,
        RecordStatus.STALE,
    ]

    # Test with replacement strategy IF_HASH_DIFFERS.
    assert not list([] >> stats_screener(ReplacementStrategy.IF_HASH_DIFFERS))
    assert list(
        [
            # Both hashes present and matching — fresh.
            make_fake_record(
                metadata={
                    "data_hash": "73ea3e7cf4f15df5e5fbd331bf8254adfae26097d0533b0e50e7777326ac20af",
                    "model_data_hash": "ee20aaa7dcb57841a6e08461c782f2ff5b62a5d5fd5ae119f274119dfcb30030",
                },
                stats=_FAKE_STATS,
            ),
            # Only data_hash present, model_data_hash is None — stale.
            make_fake_record(
                metadata={
                    "data_hash": "73ea3e7cf4f15df5e5fbd331bf8254adfae26097d0533b0e50e7777326ac20af"
                },
                stats=_FAKE_STATS,
            ),
            # Only model_data_hash present, data_hash is None — stale.
            make_fake_record(
                metadata={
                    "model_data_hash": "ee20aaa7dcb57841a6e08461c782f2ff5b62a5d5fd5ae119f274119dfcb30030"
                },
                stats=_FAKE_STATS,
            ),
            # Both hashes present and matching, but no stats — stale.
            make_fake_record(
                metadata={
                    "data_hash": "73ea3e7cf4f15df5e5fbd331bf8254adfae26097d0533b0e50e7777326ac20af",
                    "model_data_hash": "ee20aaa7dcb57841a6e08461c782f2ff5b62a5d5fd5ae119f274119dfcb30030",
                }
            ),
            # Both hashes are None — stale.
            make_fake_record(
                metadata={"data_hash": None, "model_data_hash": None},
                stats=_FAKE_STATS,
            ),
        ]
        >> stats_screener(ReplacementStrategy.IF_HASH_DIFFERS)
        >> Functor[tuple[RecordStatus, StatsRecord], RecordStatus](lambda x: x[0])
    ) == [
        RecordStatus.FRESH,
        RecordStatus.STALE,
        RecordStatus.STALE,
        RecordStatus.STALE,
        RecordStatus.STALE,
    ]
