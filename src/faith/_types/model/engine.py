# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""ModelEngine enum and EngineParams configuration."""

from dataclasses import dataclass, field
from enum import auto
from typing import Any

from dataclasses_json import DataClassJsonMixin, config

from faith._types.enums import CIEnum


class ModelEngine(CIEnum):
    """Enum representing different model engine backends."""

    OPENAI = auto()
    OPENROUTER = auto()
    VLLM = auto()
    SAGEMAKER = auto()


@dataclass(frozen=True)
class EngineParams(DataClassJsonMixin):
    """Parameters for the model backend engine."""

    engine_type: ModelEngine = field(metadata=config(decoder=ModelEngine, encoder=str))
    num_gpus: int = 1
    context_length: int = 3500
    kwargs: dict[str, Any] = field(default_factory=dict)
