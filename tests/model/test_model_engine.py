# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from faith.model.model_engine import ModelEngine


def test_model_engine_create_model() -> None:
    # Create an OpenAI-based model with a mock OpenAI api client.
    with patch("faith.model.openai.OpenAI"):
        openai_model = ModelEngine.OPENAI.create_model("fake-0.5-turbo")
        assert openai_model.name_or_path == "fake-0.5-turbo"

    with patch("faith.model.open_router.OpenRouter"):
        open_router_model = ModelEngine.OPENROUTER.create_model(
            "anthropic/claude-3.5-sonnet"
        )
        assert open_router_model.name_or_path == "anthropic/claude-3.5-sonnet"

    with patch("faith.model.sagemaker.boto3.client"):
        sagemaker_model = ModelEngine.SAGEMAKER.create_model(
            "fake-endpoint", aws_region="us-east-1"
        )
        assert sagemaker_model.name_or_path == "fake-endpoint"

    with patch("faith.model.vllm.LLM"):
        vllm_model = ModelEngine.VLLM.create_model("fake-1B-instruct")
        assert vllm_model.name_or_path == "fake-1B-instruct"
