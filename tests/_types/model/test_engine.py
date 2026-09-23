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


def test_engine_params_redacts_credential_kwargs() -> None:
    """Every credential-named kwarg serializes to a sentinel, not its value.

    One case per marker, since a dropped marker leaks that credential into every
    experiment record. Matching is on whole words, so `key` and `token` count only
    as the last one: `refresh_token` is a credential and `max_tokens` is not.
    """
    params = EngineParams(
        engine_type=ModelEngine.OPENAI,
        kwargs={
            "api_key": "sk-secret",
            "AUTH_TOKEN": "t0ken",
            "refresh_token": "r3fresh",
            "password": "hunter2",
            "client_secret": "shh",
            "gcp_credentials": "{...}",
            "base_url": "https://example.invalid/v1",
            "max_model_len": 131072,
            "max_tokens": 512,
            "tokenizer": "meta-llama/Llama-3-8B",
            "stop_token_ids": [128001],
        },
    )
    # Each key survives redaction so the record still shows a credential was set,
    # and a non-credential kwarg keeps its value whatever its type.
    assert params.to_dict()["kwargs"] == {
        "api_key": "<redacted>",
        "AUTH_TOKEN": "<redacted>",
        "refresh_token": "<redacted>",
        "password": "<redacted>",
        "client_secret": "<redacted>",
        "gcp_credentials": "<redacted>",
        "base_url": "https://example.invalid/v1",
        "max_model_len": 131072,
        "max_tokens": 512,
        "tokenizer": "meta-llama/Llama-3-8B",
        "stop_token_ids": [128001],
    }
    # Redaction is a serialization concern only: the engine still gets the real key.
    assert params.kwargs["api_key"] == "sk-secret"


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
