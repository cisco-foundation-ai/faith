# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Types use by the model backend engines."""

from dataclasses import dataclass, field
from enum import auto
from typing import Any

from dataclasses_json import DataClassJsonMixin, config

from faith._internal.algo.hash import dict_sha256
from faith._types.enums import CIStrEnum
from faith.model.model_engine import ModelEngine


class GenerationMode(CIStrEnum):
    """An enumeration of different generation modes for model outputs."""

    LOGITS = auto()
    NEXT_TOKEN = auto()
    CHAT_COMP = auto()


@dataclass(frozen=True)
class EngineParams(DataClassJsonMixin):
    """Parameters for the model backend engine."""

    engine_type: ModelEngine = field(metadata=config(decoder=ModelEngine, encoder=str))
    num_gpus: int = 1
    context_length: int = 3500
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenParams(DataClassJsonMixin):
    """Parameters for generation."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_completion_tokens: int = 500
    kwargs: dict[str, Any] = field(default_factory=dict)

    def sha256(self) -> str:
        """Compute the SHA-256 hash of the generation parameters."""
        return dict_sha256(self.to_dict())
