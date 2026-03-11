# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, NotRequired, TypeAlias, TypedDict

from dataclasses_json import DataClassJsonMixin, config

from faith._internal.algo.matching import AnswerFormat
from faith._internal.metrics.types import Labeling
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


class _ModelData(TypedDict):
    """Represents the model data associated with a log record."""

    prompt: str | ChatConversation
    answer_symbol_ids: dict[str, int]

    chat_comp: NotRequired[dict[str, Any]]
    logits: NotRequired[list[list[dict[str, Any]]]]
    next_token: NotRequired[dict[str, Any]]
    error: NotRequired[_ModelError]


@dataclass
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
    model_data: _ModelData
    stats: RecordStats | None = None


class RecordStatus(StrEnum):
    """Indicates whether a record is clean (unchanged) or dirty (new or updated)."""

    CLEAN = auto()
    DIRTY = auto()
