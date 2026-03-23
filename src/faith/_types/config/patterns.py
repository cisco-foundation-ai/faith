# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Pattern matching configuration types used by answer format matching and model response formatting."""

from dataclasses import dataclass, field
from enum import auto

from dataclasses_json import DataClassJsonMixin, config

from faith._types.enums import CIStrEnum


class AnswerFormat(CIStrEnum):
    """Enum for different ways an answer conforms to its expected format."""

    PROPER = auto()
    IMPROPER = auto()
    INFERRED = auto()
    INVALID = auto()


class Disambiguation(CIStrEnum):
    """Enum for different ways multiple matches can be disambiguated."""

    MATCH_IF_SINGULAR = auto()
    MATCH_IF_UNIQUE = auto()
    MATCH_FIRST = auto()
    MATCH_LAST = auto()
    MATCH_ALL = auto()


@dataclass(frozen=True)
class CaptureTransform(DataClassJsonMixin):
    """Configuration for transforming regex capture groups."""

    params: list[str] = field(
        default_factory=list, metadata=config(exclude=lambda x: not x)
    )
    expr: str | None = field(default=None, metadata=config(exclude=lambda x: x is None))


@dataclass(frozen=True)
class PatternDef(DataClassJsonMixin):
    """A pattern definition for matching and extracting answers from text."""

    format_type: AnswerFormat = field(
        metadata=config(encoder=str, decoder=AnswerFormat)
    )
    pattern: str = ""
    disambiguation: Disambiguation = field(
        default=Disambiguation.MATCH_IF_SINGULAR,
        metadata=config(encoder=str, decoder=Disambiguation),
    )
    capture_transform: CaptureTransform = field(default_factory=CaptureTransform)
