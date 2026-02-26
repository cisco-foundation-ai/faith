# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for reading and writing files."""

import os
import tempfile
from pathlib import Path
from typing import Any

import orjson

from faith._internal.records.types import Record


def read_json_logs(file_path: Path) -> list[Record]:
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
    """Writes `obj` to a file as a json record.

    Uses atomic write-and-rename so that a partial write (e.g. disk full,
    SIGKILL) never corrupts or truncates the target file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = orjson.dumps(
        obj,
        option=orjson.OPT_INDENT_2
        | orjson.OPT_SORT_KEYS
        | orjson.OPT_APPEND_NEWLINE
        | orjson.OPT_SERIALIZE_NUMPY,
    )
    # Temp file in the same directory ensures os.replace() is an atomic
    # same-filesystem rename. On success the rename removes the temp path;
    # on failure the context manager cleans it up.
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=file_path.parent, delete=False
    ) as tmp:
        try:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.replace(tmp.name, file_path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
