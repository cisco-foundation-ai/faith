# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for EngineParams."""

import tempfile
from pathlib import Path

import pytest

from faith._internal.io.json import read_json_file, write_as_json
from faith._types.model.engine import EngineParams, ModelEngine


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
