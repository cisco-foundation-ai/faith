# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Base model class for the model inference engines."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, cast

from transformers import PreTrainedTokenizerBase

from faith._types.model.prompt import PromptFormatter
from faith._types.records.model_response import ChatResponse, GenerationError, TokenPred

PromptList = list[str | list[dict[str, str]]]


def _is_message_list(inputs: PromptList) -> bool:
    """Check if the inputs are in a message list format."""
    return (
        isinstance(inputs, list)
        and all(isinstance(i, list) for i in inputs)
        and all(isinstance(m, dict) for i in inputs for m in i)
        and all(
            isinstance(k, str) and isinstance(v, str)
            for i in inputs
            for m in i
            if (msg := cast(dict[str, str], m))
            for k, v in msg.items()
        )
    )


def _is_string_list(inputs: PromptList) -> bool:
    """Check if the inputs are in a string list format."""
    return isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)


class BaseModel(ABC):
    """Base class for all models engine backends used to drive model inference."""

    def __init__(self, name_or_path: str) -> None:
        """Initialize the base model with the given model `name_or_path`.

        Args:
            name_or_path: the name/path that identifies the model for loading.
        """
        self._name_or_path = name_or_path

    @property
    def name_or_path(self) -> str:
        """Return the name (or path) that identifies the model for loading."""
        return self._name_or_path

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        """Return the tokenizer for the model."""
        return None

    @property
    @abstractmethod
    def supported_formats(self) -> set[PromptFormatter]:
        """Return the supported input formats for the model."""

    def logits(
        self,
        inputs: PromptList,
        **kwargs: Any,
    ) -> Iterable[list[list[TokenPred]] | GenerationError]:
        """Map each input in `inputs` to the model's next-token logits for it."""
        raise NotImplementedError("logprobs method not implemented for the model")

    def next_token(
        self,
        inputs: PromptList,
        **kwargs: Any,
    ) -> Iterable[ChatResponse | GenerationError]:
        """Map each input in `inputs` to the model's next generate token for it."""
        raise NotImplementedError("next_token method not implemented for the model")

    def query(
        self,
        inputs: PromptList,
        **kwargs: Any,
    ) -> Iterable[ChatResponse | GenerationError]:
        """Map each input in `inputs` to the model's generated response for it."""
        raise NotImplementedError("query method not implemented in the model")
