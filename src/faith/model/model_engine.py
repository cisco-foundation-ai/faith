# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""This module defines the ModelEngine enum and functions to create model instances.

The `ModelEngine` enum acts as a factory for creating instances for a given model type.
"""

from typing import Any, Callable

from faith._types.enums import CIEnum
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


class ModelEngine(CIEnum):
    """Enum representing different model engines and their factory methods.

    Each enum value corresponds to a model engine and provides a method to create
    an instance of a model using that engine. The factory methods are defined
    as static methods that take a model name and optional keyword arguments.
    """

    OPENAI = (_create_openai_model,)
    OPENROUTER = (_create_open_router_model,)
    VLLM = (_create_vllm_model,)
    SAGEMAKER = (_create_sagemaker_model,)

    @property
    def create_model_fn(self) -> Callable[..., BaseModel]:
        """Return the factory function for this model engine."""
        return self.value[0]

    def create_model(self, name_or_path: str, **kwargs: Any) -> BaseModel:
        """Create a model using the factory method associated with this value.

        Args:
            name_or_path (str): The name of the model to load with the model engine.
            **kwargs: Additional keyword arguments to pass to the model constructor.
        """
        return self.create_model_fn(name_or_path, **kwargs)
