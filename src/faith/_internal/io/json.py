# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for reading and writing files."""

from pathlib import Path
from typing import Any

import orjson


def read_json_logs(file_path: Path) -> list[dict[str, Any]]:
    """Reads a JSONL file and returns a list of dictionaries."""
    logs = read_json_file(file_path)
    assert isinstance(logs, list), f"Expected a list, got {type(logs)}"
    assert all(
        isinstance(log, dict) for log in logs
    ), "Expected all elements to be dictionaries"
    return logs


def read_json_file(file_path: Path) -> Any:
    """Reads a JSON file and returns the content."""
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as file:
        return orjson.loads(file.read())


def write_as_json(file_path: Path, obj: Any) -> None:
    """Writes `obj` to a file as a json record."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        file.write(
            orjson.dumps(
                obj,
                option=orjson.OPT_INDENT_2
                | orjson.OPT_SORT_KEYS
                | orjson.OPT_APPEND_NEWLINE
                | orjson.OPT_SERIALIZE_NUMPY,
            )
        )
