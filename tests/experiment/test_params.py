# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experiment parameter types."""

import tempfile
from pathlib import Path

import pytest

from faith._internal.io.json import read_json_file, write_as_json
from faith.experiment.params import DataSamplingParams


def test_data_sampling_params_to_dict() -> None:
    params = DataSamplingParams(sample_size=1000)
    assert params.to_dict() == {"sample_size": 1000}


@pytest.mark.parametrize(
    "params",
    [
        DataSamplingParams(sample_size=500),
        DataSamplingParams(sample_size=None),
    ],
)
def test_data_sampling_params_json_serialization(params: DataSamplingParams) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "data_sampling_params.json"
        write_as_json(file_path, params.to_dict())
        loaded_params = DataSamplingParams.from_dict(read_json_file(file_path))
        assert params == loaded_params
