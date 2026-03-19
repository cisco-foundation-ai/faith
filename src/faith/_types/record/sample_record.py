# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import auto

from dataclasses_json import DataClassJsonMixin, config

from faith._types.enums import CIStrEnum
from faith._types.record.model_record import ModelRecord
from faith._types.record.prompt_record import PromptRecord
from faith._types.record.stats import StatsRecord


@dataclass
class Metadata(DataClassJsonMixin):
    """Represents the metadata associated with a log record."""

    version: str
    data_hash: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )


@dataclass
class SampleRecord(DataClassJsonMixin):
    """Represents a log record used to track individual queries to a model."""

    metadata: Metadata
    data: PromptRecord
    model_data: ModelRecord
    stats: StatsRecord | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )


class RecordStatus(CIStrEnum):
    """Indicates whether a record is clean (unchanged) or dirty (new or updated)."""

    CLEAN = auto()
    DIRTY = auto()
