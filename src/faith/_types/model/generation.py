# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Generation mode enum and generation parameters."""

from dataclasses import dataclass, field
from enum import auto
from typing import Any

from dataclasses_json import DataClassJsonMixin

from faith._types.enums import CIStrEnum


class GenerationMode(CIStrEnum):
    """An enumeration of different generation modes for model outputs."""

    LOGITS = auto()
    NEXT_TOKEN = auto()
    CHAT_COMP = auto()


@dataclass(frozen=True)
class GenParams(DataClassJsonMixin):
    """Parameters for generation."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_completion_tokens: int = 500
    kwargs: dict[str, Any] = field(default_factory=dict)
