# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Reading benchmark trial logs from disk."""

from collections.abc import Iterable
from pathlib import Path

from faith._internal.io.json import read_logs_from_json
from faith._types.record.sample import SampleRecord


def read_trial_log(trial_log_path: Path) -> Iterable[SampleRecord]:
    """Read the trial log records from the given path."""
    if trial_log_path.exists():
        return [SampleRecord.from_dict(d) for d in read_logs_from_json(trial_log_path)]
    return []  # No records if the log file doesn't exist yet.
