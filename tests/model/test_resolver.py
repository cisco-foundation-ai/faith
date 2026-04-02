# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from faith._types.model.engine import EngineParams, ModelEngine
from faith._types.model.prompt import PromptFormatter
from faith._types.model.spec import ModelSpec
from faith.model.resolver import ResolvedModelPath


@pytest.mark.parametrize("path", ["meta-llama/Llama-2-7b", "/local/model"])
def test_resolved_model_path_non_remote(path: str) -> None:
    with ResolvedModelPath(
        ModelSpec(
            path=path,
            engine=EngineParams(engine_type=ModelEngine.OPENAI),
            prompt_format=PromptFormatter.CHAT,
        )
    ) as resolved:
        assert resolved == path


def test_resolved_model_path_remote() -> None:
    fake_local = Path("/tmp/downloaded_model")
    mock_provider = Mock(
        __enter__=Mock(return_value=fake_local), __exit__=Mock(return_value=None)
    )

    with patch(
        "faith.model.resolver.ResourceProvider", return_value=mock_provider
    ) as mock_cls:
        with ResolvedModelPath(
            ModelSpec(
                path="gs://bucket/model",
                engine=EngineParams(engine_type=ModelEngine.VLLM),
                prompt_format=PromptFormatter.CHAT,
            )
        ) as resolved:
            assert resolved == str(fake_local)
            mock_cls.assert_called_once_with("gs://bucket/model")
    mock_provider.__exit__.assert_called_once()
