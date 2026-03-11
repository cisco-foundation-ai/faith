# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, TypeAlias, TypedDict

from dataclasses_json import DataClassJsonMixin, config

from faith._internal.algo.matching import AnswerFormat
from faith._internal.metrics.types import Labeling
from faith._internal.types.flags import GenerationMode
from faith._types.records.prompt_record import PromptRecord

ChatConversation: TypeAlias = list[dict[str, str]]


class _Metadata(TypedDict):
    """Represents the metadata associated with a log record."""

    data_hash: str
    version: str


class _ModelError(TypedDict):
    """Represents an error that occurred during model inference."""

    title: str
    details: str | None


@dataclass
class ModelRecord(DataClassJsonMixin):
    """Represents the model data associated with a log record."""

    prompt: str | ChatConversation
    answer_symbol_ids: dict[str, int]

    logits: list[list[dict[str, Any]]] | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    next_token: dict[str, Any] | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    chat_comp: dict[str, Any] | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    error: _ModelError | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )

    def reset_to_mode(self, mode: GenerationMode) -> None:
        """Resets the model data to only include response fields for the specified generation mode."""
        if mode != GenerationMode.LOGITS:
            self.logits = None
        if mode != GenerationMode.NEXT_TOKEN:
            self.next_token = None
        if mode != GenerationMode.CHAT_COMPLETION:
            self.chat_comp = None
        self.error = None


@dataclass(frozen=True)
class RecordStats(DataClassJsonMixin):
    """Statistics computed for a single record by a log grader."""

    label: Labeling | None
    prediction: Labeling | None
    answer_format: AnswerFormat = field(
        metadata=config(decoder=AnswerFormat.from_string, encoder=str)
    )
    subject: str | None = None
    log_probs: dict[str, float] | None = None
    num_output_tokens: int | None = None
    max_token_halt: bool | None = None
    scores: dict[str, Any] | None = None


@dataclass
class Record(DataClassJsonMixin):
    """Represents a log record used to track individual queries to a model."""

    metadata: _Metadata
    data: PromptRecord
    model_data: ModelRecord
    stats: RecordStats | None = None


class RecordStatus(StrEnum):
    """Indicates whether a record is clean (unchanged) or dirty (new or updated)."""

    CLEAN = auto()
    DIRTY = auto()
