# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum, auto
from typing import Any, NotRequired, TypeAlias, TypedDict

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


class Record(TypedDict):
    """Represents a log record used to track individual queries to a model."""

    metadata: _Metadata
    data: PromptRecord
    model_data: _ModelData
    stats: dict[str, Any] | None


class RecordStatus(StrEnum):
    """Indicates whether a record is clean (unchanged) or dirty (new or updated)."""

    CLEAN = auto()
    DIRTY = auto()
