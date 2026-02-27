# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for reading and writing records."""

from pathlib import Path

from faith._internal.io.json import read_json_file
from faith._internal.records.types import Record


def load_records_from_json(file_path: Path) -> list[Record]:
    """Reads a JSONL file and returns a list of dictionaries."""
    logs = read_json_file(file_path)
    assert isinstance(logs, list), f"Expected a list, got {type(logs)}"
    assert all(
        isinstance(log, dict) for log in logs
    ), "Expected all elements to be dictionaries"
    return logs
