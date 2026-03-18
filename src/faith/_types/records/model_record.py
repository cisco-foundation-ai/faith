# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin, config

from faith._types.model.generation import GenerationMode
from faith._types.records.model_response import (
    ChatConversation,
    ChatResponse,
    GenerationError,
    TokenPred,
)


@dataclass
class ModelRecord(DataClassJsonMixin):
    """Represents the model data associated with a log record."""

    prompt: str | ChatConversation
    answer_symbol_ids: dict[str, int]

    logits: list[list[TokenPred]] | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    next_token: ChatResponse | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    chat_comp: ChatResponse | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    error: GenerationError | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )

    def reset_to_mode(self, mode: GenerationMode) -> None:
        """Resets the model data to only include response fields for the specified generation mode."""
        if mode != GenerationMode.LOGITS:
            self.logits = None
        if mode != GenerationMode.NEXT_TOKEN:
            self.next_token = None
        if mode != GenerationMode.CHAT_COMP:
            self.chat_comp = None
        self.error = None
