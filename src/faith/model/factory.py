# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Factory function for creating model instances by engine type."""

from typing import Any

from faith._types.model.engine import ModelEngine
from faith.model.base import BaseModel

# The following functions create instances of different model types and they
# allow us to only load the necessary model libraries on model creation.
# This is useful for reducing the initial load time of the script particularly for
# auto-complete purposes.


def _create_openai_model(name_or_path: str, **kwargs: Any) -> BaseModel:
    # We disable the import-outside-toplevel pylint rule here because
    # the imports required each model type are only installed as package extras
    # to allow for a smaller install footprint.
    # pylint: disable=import-outside-toplevel
    from faith.model.openai import OpenAIModel

    return OpenAIModel(name_or_path, **kwargs)


def _create_open_router_model(name_or_path: str, **kwargs: Any) -> BaseModel:
    # We disable the import-outside-toplevel pylint rule here because
    # the imports required each model type are only installed as package extras
    # to allow for a smaller install footprint.
    # pylint: disable=import-outside-toplevel
    from faith.model.open_router import OpenRouterModel

    return OpenRouterModel(name_or_path, **kwargs)


def _create_vllm_model(name_or_path: str, **kwargs: Any) -> BaseModel:
    # We disable the import-outside-toplevel pylint rule here because
    # the imports required each model type are only installed as package extras
    # to allow for a smaller install footprint.
    # pylint: disable=import-outside-toplevel
    from faith.model.vllm import VLLMModel

    return VLLMModel(name_or_path, **kwargs)


def _create_sagemaker_model(name_or_path: str, **kwargs: Any) -> BaseModel:
    # We disable the import-outside-toplevel pylint rule here because
    # the imports required each model type are only installed as package extras
    # to allow for a smaller install footprint.
    # pylint: disable=import-outside-toplevel
    from faith.model.sagemaker import SageMakerModel

    return SageMakerModel(name_or_path, **kwargs)


_ENGINE_FACTORIES: dict[ModelEngine, Any] = {
    ModelEngine.OPENAI: _create_openai_model,
    ModelEngine.OPENROUTER: _create_open_router_model,
    ModelEngine.VLLM: _create_vllm_model,
    ModelEngine.SAGEMAKER: _create_sagemaker_model,
}


def create_model(engine: ModelEngine, name_or_path: str, **kwargs: Any) -> BaseModel:
    """Create a model instance for the given engine type.

    Model libraries are imported lazily to reduce initial load time and allow
    a smaller install footprint when only a subset of engines is needed.

    Args:
        engine: The model engine backend to use.
        name_or_path: The name or path that identifies the model for loading.
        **kwargs: Additional keyword arguments to pass to the model constructor.
    """
    return _ENGINE_FACTORIES[engine](name_or_path, **kwargs)
