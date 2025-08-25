# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Types use by the model backend engines."""
from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import DataClassJsonMixin, config

from faith._internal.algo.hash import dict_sha256
from faith.model.model_engine import ModelEngine


@dataclass
class EngineParams(DataClassJsonMixin):
    """Parameters for the model backend engine."""

    engine_type: ModelEngine = field(
        metadata=config(decoder=ModelEngine.from_string, encoder=str)
    )
    num_gpus: int
    context_length: int
    kwargs: dict[str, Any]


@dataclass
class GenParams(DataClassJsonMixin):
    """Parameters for generation."""

    temperature: float
    top_p: float
    max_completion_tokens: int
    kwargs: dict[str, Any]

    def sha256(self) -> str:
        """Compute the SHA-256 hash of the generation parameters."""
        return dict_sha256(self.to_dict())
