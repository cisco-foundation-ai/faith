# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Reading and writing benchmark trial logs from disk."""

from collections.abc import Iterable
from pathlib import Path

from faith._internal.io.json import read_logs_from_json, write_as_json
from faith._internal.io.logging import LoggingTransform
from faith._internal.iter.transform import IsoTransform, Transform
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

    The batch query-side tail used by QA: the model-data hash lets
    the grader skip unchanged answers, the sort undoes the completion-order
    scrambling introduced by the Mux, and the log write persists records for the
    eval phase. Returns a Transform (records in, records out), so callers still
    drive execution with a terminal reducer. Agentic uses checkpoint_records to
    avoid buffering the entire trial before saving its completed tasks.
    """
    return (
        HashModelDataTransform()
        | SortByTransform[int]("data", "benchmark_sample_index")
        | LoggingTransform[SampleRecord](trial_log_path)
    )


class _CheckpointTrialLogTransform(IsoTransform[SampleRecord]):
    """Atomically replace completed records while retaining untouched resume data."""

    def __init__(self, path: Path, existing: Iterable[SampleRecord]):
        self._path = path
        self._existing = {
            record.data.benchmark_sample_index: record for record in existing
        }

    def __call__(self, src: Iterable[SampleRecord]) -> Iterable[SampleRecord]:
        """Write before yielding, and prune unselected records only on completion."""
        checkpoint = dict(self._existing)
        selected: dict[int, SampleRecord] = {}
        for record in src:
            index = record.data.benchmark_sample_index
            selected[index] = record
            checkpoint[index] = record
            # Sort the accumulated snapshot, not the upstream execution stream.
            # Never catch write errors: the caller must stop before another task.
            write_as_json(self._path, [checkpoint[i] for i in sorted(checkpoint)])
            yield record
        # A successful nonempty run replaces the log with its selected task set,
        # just like LoggingTransform. Empty selections leave the old log alone.
        # An interrupted run never reaches this pruning step.
        if selected and selected.keys() != checkpoint.keys():
            write_as_json(self._path, [selected[i] for i in sorted(selected)])


def checkpoint_records(
    trial_log_path: Path, existing_records: Iterable[SampleRecord]
) -> Transform[SampleRecord, SampleRecord]:
    """Hash and atomically checkpoint each complete agentic record before yielding.

    Requires an execution source that cannot run ahead of its consumer. Keeps the
    existing JSON-list format and prior records until selection completes; this
    does not upload to remote storage or checkpoint a task still in progress.
    """
    return HashModelDataTransform() | _CheckpointTrialLogTransform(
        trial_log_path, existing_records
    )
