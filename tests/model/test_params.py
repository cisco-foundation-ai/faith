# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model backend types."""

import tempfile
from pathlib import Path

import pytest

from faith._internal.io.json import read_json_file, write_as_json
from faith.model.model_engine import ModelEngine
from faith.model.params import EngineParams, GenParams


def test_engine_params_to_dict() -> None:
    """Test the EngineParams dataclass."""
    params = EngineParams(
        engine_type=ModelEngine.VLLM,
        num_gpus=2,
        context_length=2048,
        kwargs={"max_model_len": 131072, "enable_prefix_caching": False},
    )
    assert params.to_dict() == {
        "engine_type": "vllm",
        "num_gpus": 2,
        "context_length": 2048,
        "kwargs": {"max_model_len": 131072, "enable_prefix_caching": False},
    }


@pytest.mark.parametrize(
    "params",
    [
        EngineParams(
            engine_type=ModelEngine.VLLM,
            num_gpus=256,
            context_length=1024,
            kwargs={"max_model_len": 131072, "enable_prefix_caching": True},
        ),
        EngineParams(
            engine_type=ModelEngine.OPENAI, num_gpus=0, context_length=4096, kwargs={}
        ),
    ],
)
def test_engine_params_json_serialization(params: EngineParams) -> None:
    """Test the JSON serialization & deserialization of EngineParams."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "engine_params.json"
        write_as_json(file_path, params.to_dict())
        loaded_params = params.from_dict(read_json_file(file_path))
        assert params == loaded_params


def test_gen_params_sha256() -> None:
    """Test the SHA-256 hash of GenParams."""
    params = GenParams(
        temperature=0.5,
        top_p=0.95,
        max_completion_tokens=150,
        kwargs={"key1": "value1", "key2": 42},
    )
    assert (
        params.sha256()
        == "033ac5111b83dde7c8464243d49205b55cdcd1d85d4abb151af24bed25b10718"
    )


def test_gen_params_to_dict() -> None:
    """Test the GenParams dataclass."""
    params = GenParams(
        temperature=0.7,
        top_p=0.9,
        max_completion_tokens=100,
        kwargs={"some_key": 10, "other_key": "foo"},
    )
    assert params.to_dict() == {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_completion_tokens": 100,
        "kwargs": {"some_key": 10, "other_key": "foo"},
    }


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
