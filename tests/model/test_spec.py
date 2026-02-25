# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model config YAML loader."""

from pathlib import Path

import pytest
from dacite.exceptions import MissingValueError

from faith.model.base import ReasoningSpec
from faith.model.model_engine import ModelEngine
from faith.model.params import EngineParams, GenParams
from faith.model.spec import ModelSpec

_CONFIGS_DIR = Path(__file__).parent / "testdata" / "configs"


def test_full_config() -> None:
    """All fields specified — assert exact values propagate."""
    assert ModelSpec.from_file(_CONFIGS_DIR / "full_config.yaml") == ModelSpec(
        path="meta-llama/Llama-2-7b",
        engine=EngineParams(
            engine_type=ModelEngine.VLLM,
            num_gpus=4,
            context_length=4096,
            kwargs={"max_num_batched_tokens": 8192},
        ),
        name="llama2-7b",
        tokenizer="/path/to/tokenizer",
        reasoning=ReasoningSpec(start_delimiter="<think>", end_delimiter="</think>"),
        response_pattern="Answer: (.*)",
        generation=GenParams(
            temperature=0.7,
            top_p=0.95,
            max_completion_tokens=1024,
            kwargs={"repetition_penalty": 1.1},
        ),
    )


def test_minimal_config_applies_defaults() -> None:
    """Only required fields — assert all defaults match CLI defaults."""
    assert ModelSpec.from_file(_CONFIGS_DIR / "minimal_config.yaml") == ModelSpec(
        path="gpt-4o",
        name="gpt-4o",
        engine=EngineParams(
            engine_type=ModelEngine.OPENAI,
        ),
    )


def test_no_generation_section() -> None:
    """Missing generation section entirely — assert gen param defaults."""
    assert ModelSpec.from_file(_CONFIGS_DIR / "no_generation.yaml") == ModelSpec(
        path="meta-llama/Llama-2-7b",
        engine=EngineParams(
            engine_type=ModelEngine.VLLM,
            num_gpus=2,
            context_length=2048,
        ),
        name="llama2",
    )


def test_missing_model_path_raises() -> None:
    """Missing model.path — should raise ValueError."""
    with pytest.raises(MissingValueError, match='missing value for field "path"'):
        ModelSpec.from_file(_CONFIGS_DIR / "missing_path.yaml")


def test_missing_engine_section_raises() -> None:
    """Missing engine section entirely — should raise ValueError."""
    with pytest.raises(MissingValueError, match='missing value for field "engine"'):
        ModelSpec.from_file(_CONFIGS_DIR / "missing_engine.yaml")


def test_missing_engine_type_raises() -> None:
    """Missing engine.type — should raise ValueError."""
    with pytest.raises(
        MissingValueError, match='missing value for field "engine.engine_type"'
    ):
        ModelSpec.from_file(_CONFIGS_DIR / "missing_engine_type.yaml")


def test_composed_config_with_from_directive() -> None:
    """Engine loaded via !from directive — assert merge works."""
    assert ModelSpec.from_file(_CONFIGS_DIR / "composed_config.yaml") == ModelSpec(
        path="meta-llama/Llama-2-13b",
        name="meta-llama_Llama-2-13b",
        engine=EngineParams(
            engine_type=ModelEngine.VLLM,
            num_gpus=2,
            context_length=4096,
        ),
    )
