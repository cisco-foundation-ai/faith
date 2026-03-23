# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for GenParams."""

import tempfile
from pathlib import Path

import pytest

from faith._internal.algo.hash import dict_sha256
from faith._internal.io.json import read_json_file, write_as_json
from faith._types.model.generation import GenParams


def test_gen_params_to_dict() -> None:
    """Test the GenParams dataclass."""
    dict_repr = GenParams(
        temperature=0.5,
        top_p=0.95,
        max_completion_tokens=150,
        kwargs={"key1": "value1", "key2": 42},
    ).to_dict()
    assert dict_repr == {
        "temperature": 0.5,
        "top_p": 0.95,
        "max_completion_tokens": 150,
        "kwargs": {"key1": "value1", "key2": 42},
    }
    assert (
        dict_sha256(dict_repr)
        == "033ac5111b83dde7c8464243d49205b55cdcd1d85d4abb151af24bed25b10718"
    ), f"Hash of gen params has changed: {dict_sha256(dict_repr)}"


@pytest.mark.parametrize(
    "params",
    [
        GenParams(
            temperature=0.1,
            top_p=0.5,
            max_completion_tokens=50,
            kwargs={},
        ),
        GenParams(
            temperature=0.9,
            top_p=0.95,
            max_completion_tokens=200,
            kwargs={"key": 37},
        ),
    ],
)
def test_gen_params_json_serialization(params: GenParams) -> None:
    """Test the JSON serialization & deserialization of GenParams."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "gen_params.json"
        write_as_json(file_path, params.to_dict())
        loaded_params = params.from_dict(read_json_file(file_path))
        assert params == loaded_params
