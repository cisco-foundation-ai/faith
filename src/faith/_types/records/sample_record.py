# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TypedDict

from dataclasses_json import DataClassJsonMixin

from faith._types.records.model_record import ModelRecord
from faith._types.records.prompt_record import PromptRecord
from faith._types.records.stats_record import StatsRecord


class _Metadata(TypedDict):
    """Represents the metadata associated with a log record."""

    data_hash: str
    version: str


@dataclass
class SampleRecord(DataClassJsonMixin):
    """Represents a log record used to track individual queries to a model."""

    metadata: _Metadata
    data: PromptRecord
    model_data: ModelRecord
    stats: StatsRecord | None = None


class RecordStatus(StrEnum):
    """Indicates whether a record is clean (unchanged) or dirty (new or updated)."""

    CLEAN = auto()
    DIRTY = auto()
