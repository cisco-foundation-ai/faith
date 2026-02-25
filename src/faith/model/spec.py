# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Complete model specification bundling path, engine, and generation parameters."""

from dataclasses import dataclass, field
from pathlib import Path

from dacite import Config, from_dict
from dataclasses_json import DataClassJsonMixin

from faith._internal.io.paths import canonical_segment
from faith._internal.io.yaml import read_extended_yaml_file
from faith.model.base import ReasoningSpec
from faith.model.model_engine import ModelEngine
from faith.model.params import EngineParams, GenParams


@dataclass
class ModelSpec(DataClassJsonMixin):
    """A complete model specification.

    Bundles a model's annotated path with its specific engine and generation
    parameters, enabling per-model configuration when using --model-configs.
    """

    path: str
    engine: EngineParams
    name: str = ""
    reasoning: ReasoningSpec | None = None
    response_pattern: str | None = None
    tokenizer: str | None = None
    generation: GenParams = field(default_factory=GenParams)

    def __post_init__(self) -> None:
        """Validate the ModelSpec after initialization."""
        assert self.path, "Model path must be a non-empty string."
        if not self.name:
            self.name = canonical_segment(self.path)

    @staticmethod
    def from_file(config_path: Path) -> "ModelSpec":
        """Load a ModelSpec from a YAML configuration file."""
        model_spec_dict = read_extended_yaml_file(config_path).get("model", {})
        assert isinstance(
            model_spec_dict, dict
        ), f"Model config '{config_path}' must be a YAML mapping."

        return from_dict(
            data_class=ModelSpec,
            data=model_spec_dict,
            config=Config(type_hooks={ModelEngine: ModelEngine.from_string}),
        )
