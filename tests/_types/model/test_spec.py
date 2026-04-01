# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelSpec."""

import pytest

from faith._types.model.engine import EngineParams, ModelEngine
from faith._types.model.prompt import PromptFormatter
from faith._types.model.spec import ModelSpec


@pytest.mark.parametrize(
    "path, expected",
    [
        ("gs://bucket/model", True),
        ("gs://bucket/path/to/model", True),
        ("meta-llama/Llama-2-7b", False),
        ("gpt-4o-mini-2024-07-18", False),
        ("/local/path/to/model", False),
        ("./relative/path", False),
    ],
)
def test_model_spec_is_remote(path: str, expected: bool) -> None:
    """is_remote should be True only for GCS paths."""
    spec = ModelSpec(
        path=path,
        engine=EngineParams(engine_type=ModelEngine.VLLM),
        prompt_format=PromptFormatter.CHAT,
    )
    assert spec.is_remote is expected
