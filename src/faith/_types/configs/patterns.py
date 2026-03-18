# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Pattern matching configuration types used by answer format matching and model response formatting."""

from dataclasses import dataclass, field
from enum import auto

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
class CaptureTransform:
    """Configuration for transforming regex capture groups."""

    params: list[str] = field(default_factory=list)
    expr: str | None = None


@dataclass(frozen=True)
class PatternDef:
    """A pattern definition for matching and extracting answers from text."""

    format_type: AnswerFormat
    pattern: str = ""
    disambiguation: Disambiguation = Disambiguation.MATCH_IF_SINGULAR
    capture_transform: CaptureTransform = field(default_factory=CaptureTransform)
