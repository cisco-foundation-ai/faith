# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Reading and writing benchmark trial logs from disk."""

from collections.abc import Iterable
from pathlib import Path

from faith._internal.io.json import read_logs_from_json
from faith._internal.io.logging import LoggingTransform
from faith._internal.iter.transform import Transform
from faith._types.record.sample import SampleRecord
from faith.record_pipelines.hashing import HashModelDataTransform
from faith.record_pipelines.sorting import SortByTransform


def read_trial_log(trial_log_path: Path) -> Iterable[SampleRecord]:
    """Read the trial log records from the given path."""
    if trial_log_path.exists():
        return [SampleRecord.from_dict(d) for d in read_logs_from_json(trial_log_path)]
    return []  # No records if the log file doesn't exist yet.


def persist_records(trial_log_path: Path) -> Transform[SampleRecord, SampleRecord]:
    """Hash each record's model data, restore input order, then write to the trial log.

    The shared query-side tail for every trial runner: the model-data hash lets
    the grader skip unchanged answers, the sort undoes the completion-order
    scrambling introduced by the Mux, and the log write persists records for the
    eval phase. Returns a Transform (records in, records out), so callers still
    drive execution with a terminal reducer.
    """
    return (
        HashModelDataTransform()
        | SortByTransform[int]("data", "benchmark_sample_index")
        | LoggingTransform[SampleRecord](trial_log_path)
    )
