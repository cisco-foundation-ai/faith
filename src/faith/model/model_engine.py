# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""This module defines the ModelEngine enum and functions to create model instances.

The `ModelEngine` enum acts as a factory for creating instances for a given model type.
"""
from enum import Enum
from typing import Any, Callable

from faith.model.base import BaseModel


# The following functions create instances of different model types and they
# allow us to only load the necessary model libraries on model creation.abs
# This is useful for reducing the initial load time of the script particularly for
# auto-complete purposes.
def _create_openai_model(name_or_path: str, **kwargs: Any) -> BaseModel:
    # We disable the import-outside-toplevel pylint rule here because
    # the imports required each model type are only installed as package extras
    # to allow for a smaller install footprint.
    # pylint: disable=import-outside-toplevel
    from faith.model.openai import OpenAIModel

    # pylint: enable=import-outside-toplevel

    return OpenAIModel(name_or_path, **kwargs)


def _create_vllm_model(name_or_path: str, **kwargs: Any) -> BaseModel:
    # We disable the import-outside-toplevel pylint rule here because
    # the imports required each model type are only installed as package extras
    # to allow for a smaller install footprint.
    # pylint: disable=import-outside-toplevel
    from faith.model.vllm import VLLMModel

    # pylint: enable=import-outside-toplevel

    return VLLMModel(name_or_path, **kwargs)


class ModelEngine(Enum):
    """Enum representing different model engines and their factory methods.

    Each enum value corresponds to a model engine and provides a method to create
    an instance of a model using that engine. The factory methods are defined
    as static methods that take a model name and optional keyword arguments.
    """

    OPENAI = (_create_openai_model,)
    VLLM = (_create_vllm_model,)

    def __init__(self, create_model_fn: Callable[[str], BaseModel]) -> None:
        """Initialize the ModelEngine enum with its corresponding factory function."""
        self._create_model_fn = create_model_fn

    def __str__(self) -> str:
        """Return the string representation of the enum value."""
        return self.name.lower()

    @staticmethod
    def from_string(name: str) -> "ModelEngine":
        """Convert a string to the corresponding ModelEngine enum."""
        try:
            return ModelEngine[name.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown model type: {name}") from e

    def create_model(self, name_or_path: str, **kwargs: Any) -> BaseModel:
        """Create a model using the factory method associated with this value.

        Args:
            name_or_path (str): The name of the model to load with the model engine.
            **kwargs: Additional keyword arguments to pass to the model constructor.
        """
        return self._create_model_fn(name_or_path, **kwargs)
