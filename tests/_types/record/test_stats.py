# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the StatsRecord record type."""

import pytest

from faith._types.config.patterns import AnswerFormat
from faith._types.record.stats import Labeling, StatsRecord


@pytest.mark.parametrize(
    "value",
    [
        "flag{x}",
        ["a.c", "b.c"],
        42,
        3.14,
        True,
        {"file": "a.c", "line": 10},
        [{"file": "a.c"}, {"file": "b.c"}],
        None,
    ],
)
def test_stats_record_round_trips_labeling_variants(value: Labeling) -> None:
    # Labeling is any JSON value: every variant must survive to_dict/from_dict.
    record = StatsRecord(
        label=value, prediction=value, answer_format=AnswerFormat.PROPER
    )
    assert StatsRecord.from_dict(record.to_dict()) == record
