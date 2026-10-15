# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Generation mode enum and generation parameters."""

from dataclasses import dataclass, field
from enum import auto
from typing import Any

from dataclasses_json import DataClassJsonMixin, config

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
    kwargs: dict[str, Any] = field(
        default_factory=dict, metadata=config(exclude=lambda x: not x)
    )

    def as_request_params(self) -> dict[str, Any]:
        """These params as keyword arguments for `chat.completions.create`.

        `max_completion_tokens` is the reasoning-model spelling of `max_tokens`, and
        -1 is the "omit, use the API default" sentinel (see `APIBasedModel.query`);
        forwarding it verbatim would make the API reject reasoning configs like o3.
        """
        params: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            **self.kwargs,
        }
        if self.max_completion_tokens > 0:
            params["max_completion_tokens"] = self.max_completion_tokens
        return params
