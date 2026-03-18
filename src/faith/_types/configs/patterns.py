# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Pattern matching configuration types used by answer format matching and model response formatting."""

from dataclasses import dataclass, field
from enum import Enum


class AnswerFormat(Enum):
    """Enum for different ways an answer conforms to its expected format."""

    PROPER = "proper"
    IMPROPER = "improper"
    INFERRED = "inferred"
    INVALID = "invalid"

    def __str__(self) -> str:
        """Return the string representation of the enum."""
        return self.value

    @staticmethod
    def from_string(s: str) -> "AnswerFormat":
        """Convert a string to an AnswerFormat enum."""
        try:
            return AnswerFormat[s.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown answer format: {s}") from e


class Disambiguation(Enum):
    """Enum for different ways multiple matches can be disambiguated."""

    MATCH_IF_SINGULAR = "match_if_singular"
    MATCH_IF_UNIQUE = "match_if_unique"
    MATCH_FIRST = "match_first"
    MATCH_LAST = "match_last"
    MATCH_ALL = "match_all"

    def __str__(self) -> str:
        """Return the string representation of the enum."""
        return self.value

    @staticmethod
    def from_string(s: str) -> "Disambiguation":
        """Convert a string to a Disambiguation enum."""
        try:
            return Disambiguation[s.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown disambiguation: {s}") from e


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
