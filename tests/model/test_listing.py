# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pytest_unordered import unordered

from faith._internal.io.resources import models_root
from faith.model.listing import (
    choice_to_model,
    find_models,
    model_choices,
)

ALL_INSTRUCT = [
    "instruct/deephat-v1-7B",
    "instruct/falcon3-10B-instruct",
    "instruct/falcon3-1B-instruct",
    "instruct/falcon3-3B-instruct",
    "instruct/falcon3-7B-instruct",
    "instruct/foundation-sec-1.1-8B-instruct",
    "instruct/foundation-sec-8B-instruct",
    "instruct/gemma-3-12b-it",
    "instruct/gemma-3-1b-it",
    "instruct/gemma-3-27b-it",
    "instruct/gemma-3-4b-it",
    "instruct/gpt-4.1-2025-04-14",
    "instruct/gpt-4o-mini-2024-07-18",
    "instruct/lily-cybersecurity-7B-v0.2",
    "instruct/llama-3.1-70B-instruct",
    "instruct/llama-3.1-8B-instruct",
    "instruct/llama-4-maverick-17B-128E-instruct-FP8",
    "instruct/llama-4-scout-17B-16E-instruct",
    "instruct/llama-primus-merged",
    "instruct/llama-primus-reasoning",
    "instruct/mistral-7B-instruct-v0.2",
    "instruct/mistral-7B-instruct-v0.3",
    "instruct/mistral-nemo-instruct-2407",
    "instruct/phi-4",
    "instruct/phi-4-mini-instruct",
    "instruct/qwen2.5-1.5B-instruct",
    "instruct/qwen2.5-14B-instruct",
    "instruct/qwen2.5-32B-instruct",
    "instruct/qwen2.5-3B-instruct",
    "instruct/qwen2.5-72B-instruct",
    "instruct/qwen2.5-7B-instruct",
    "instruct/whiterabbitneo-2.5-qwen-2.5-coder-7B",
]

ALL_REASONING = [
    "reasoning/foundation-sec-8B-reasoning",
    "reasoning/gpt-5-nano-2025-08-07",
    "reasoning/gpt-oss-120b",
    "reasoning/gpt-oss-20b",
    "reasoning/o3-2025-04-16",
]

ALL_MODELS = ALL_INSTRUCT + ALL_REASONING


def model_subpath(sub_paths: list[str]) -> list[Path]:
    """Helper to convert sub-path strings to full model config paths."""
    return [models_root() / f"{p}.yaml" for p in sub_paths]


def test_model_choices() -> None:
    assert model_choices() == unordered(ALL_MODELS)


def test_choice_to_model_packaged() -> None:
    assert choice_to_model("instruct/phi-4") == models_root() / "instruct/phi-4.yaml"
    assert (
        choice_to_model("reasoning/o3-2025-04-16")
        == models_root() / "reasoning/o3-2025-04-16.yaml"
    )


def test_choice_to_model_local_path() -> None:
    assert choice_to_model("/tmp/my-model.yaml") == Path("/tmp/my-model.yaml")
    assert choice_to_model("./relative/model.yaml") == Path("./relative/model.yaml")


def test_find_models() -> None:
    assert find_models(models_root()) == unordered(model_subpath(ALL_MODELS))
