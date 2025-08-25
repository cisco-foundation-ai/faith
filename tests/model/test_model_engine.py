# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from faith.model.model_engine import ModelEngine


def test_model_engine_strings() -> None:
    assert ModelEngine.from_string(str(ModelEngine.OPENAI)) == ModelEngine.OPENAI
    assert ModelEngine.from_string(str(ModelEngine.VLLM)) == ModelEngine.VLLM

    with pytest.raises(ValueError, match="Unknown model type: unknown"):
        ModelEngine.from_string("unknown")


def test_model_engine_create_model() -> None:
    # Create an OpenAI-based model with a mock OpenAI api client.
    with patch("faith.model.openai.OpenAI"):
        openai_model = ModelEngine.OPENAI.create_model("fake-0.5-turbo")
        assert openai_model.name_or_path == "fake-0.5-turbo"

    with patch("faith.model.vllm.LLM"):
        vllm_model = ModelEngine.VLLM.create_model("fake-1B-instruct")
        assert vllm_model.name_or_path == "fake-1B-instruct"
