# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from pathlib import Path

from faith._internal.io.json import read_logs_from_json
from faith._internal.io.logging import LoggingTransform
from faith._internal.iter.common import GetAttrTransform
from faith._internal.iter.mux import MuxTransform
from faith._internal.iter.transform import IdentityTransform
from faith._types.record.sample import RecordStatus, ReplacementStrategy, SampleRecord
from faith._types.record.stats import StatsRecord
from faith.benchmark.grading.log_grader import LogGrader
from faith.record_pipelines.screening import stats_screener
from faith.record_pipelines.sorting import SortByTransform


def grade_trial_records(
    grader: LogGrader,
    trial_path: Path,
    *,
    replacement_strategy: ReplacementStrategy = ReplacementStrategy.NEVER,
) -> Iterable[StatsRecord | None]:
    """Grade an experiment's trial logs with the given grader, yielding its StatsRecords."""
    return (
        [SampleRecord.from_dict(d) for d in read_logs_from_json(trial_path)]
        >> stats_screener(replacement_strategy)
        >> MuxTransform(
            {
                RecordStatus.FRESH: IdentityTransform[SampleRecord](),
                RecordStatus.STALE: grader,
            }
        )
        >> SortByTransform[int]("data", "benchmark_sample_index")
        >> LoggingTransform[SampleRecord](trial_path)
        >> GetAttrTransform[SampleRecord, StatsRecord | None]("stats")
    )
