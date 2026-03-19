# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Complete model specification bundling path, engine, and generation parameters."""

from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin, config

from faith._types.model.engine import EngineParams
from faith._types.model.generation import GenParams
from faith._types.model.naming import canonical_segment
from faith._types.model.prompt import PromptFormatter


@dataclass(frozen=True)
class Reasoning(DataClassJsonMixin):
    """Delimiters used to denote reasoning steps in the model's output."""

    start_delimiter: str | list[int]
    end_delimiter: str | list[int]


@dataclass(frozen=True)
class ModelSpec(DataClassJsonMixin):
    """A complete model specification.

    Bundles a model's annotated path with its specific engine and generation
    parameters, enabling per-model configuration when using --model-configs.
    """

    path: str
    engine: EngineParams
    prompt_format: PromptFormatter = field(
        metadata=config(encoder=str, decoder=PromptFormatter)
    )
    name: str = ""
    reasoning: Reasoning | None = None
    response_pattern: str | None = None
    tokenizer: str | None = None
    generation: GenParams = field(default_factory=GenParams)

    def __post_init__(self) -> None:
        """Validate the ModelSpec after initialization."""
        assert self.path, "Model path must be a non-empty string."
        if not self.name:
            # Bypass frozen=True to set the name based on the path if not provided.
            object.__setattr__(self, "name", canonical_segment(self.path))
