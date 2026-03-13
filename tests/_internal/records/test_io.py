# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from faith._internal.io.json import write_as_json
from faith._internal.records.io import load_records_from_json
from faith._types.records.sample_record import SampleRecord


@pytest.mark.parametrize(
    "test_data",
    [
        [],
        [{"name": "test_record_1", "value": 1}, {"name": "test_record_2", "value": 2}],
    ],
)
def test_load_records_from_json(test_data: list[SampleRecord]) -> None:
    """Test loading a JSON log file of records."""
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_path = Path(temp_file.name)
        write_as_json(temp_path, test_data)
        assert load_records_from_json(temp_path) == test_data
