# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for reading and writing records."""

from pathlib import Path
from typing import Any

from faith._internal.io.json import read_json_file


def load_records_from_json(file_path: Path) -> list[dict[str, Any]]:
    """Reads a JSONL file and returns a list of dictionaries."""
    logs = read_json_file(file_path)
    assert isinstance(logs, list), f"Expected a list, got {type(logs)}"
    assert all(
        isinstance(log, dict) for log in logs
    ), "Expected all elements to be dictionaries"
    return logs
