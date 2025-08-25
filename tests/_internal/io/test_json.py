# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from faith._internal.io.json import read_json_file, read_json_logs, write_as_json


def test_read_json_file() -> None:
    """Test reading a JSON file."""
    test_path = Path(__file__).parent / "testdata" / "record.json"
    expected_data = {
        "name": "test_record",
        "records": [
            {
                "id": 123,
                "name": "record1",
                "description": "This is the first record.",
            },
            {
                "id": 456,
                "name": "record2",
                "description": "This is the second record.",
            },
        ],
    }
    data = read_json_file(test_path)
    assert data == expected_data, f"Expected {expected_data}, but got {data}"


def test_read_json_file_nonexistent() -> None:
    """Test reading a non-existent JSON file."""
    with pytest.raises(FileNotFoundError):
        read_json_file(Path("non_existent_file.json"))


def test_read_json_logs() -> None:
    """Test reading a JSON logs file."""
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_path = Path(temp_file.name)
        test_data = [
            {"name": "test_record_1", "value": 1},
            {"name": "test_record_2", "value": 2},
        ]
        write_as_json(temp_path, test_data)

        logs = read_json_logs(temp_path)
        assert logs == test_data, f"Expected {test_data}, but got {logs}"


@pytest.mark.parametrize(
    "test_data",
    [
        {"key": "value"},
        {"key": ["value1", "value2"]},
        {"key": {"subkey": "subvalue"}},
        [
            {"key": {"subkey": "subvalue"}},
            {"key": ["value1", "value2"]},
            {"key": 1.0},
            {"key": np.float64(1.0)},
        ],
        {"key": 123},
        {"key": True},
    ],
)
def test_write_as_json(test_data: Any) -> None:
    """Test the write_as_json function."""
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_path = Path(temp_file.name)

        # Write the test data to a JSON file
        write_as_json(temp_path, test_data)

        # Read the data back from the file
        read_data = read_json_file(temp_path)

        # Assert that the written and read data are equal
        assert read_data == test_data, f"Expected {test_data}, but got {read_data}"
