# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for matching and processing answer formats."""
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, TypeVar, cast

from faith._internal.parsing.expr import evaluate_expr


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


class MatchDisambiguation(Enum):
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
    def from_string(s: str) -> "MatchDisambiguation":
        """Convert a string to a MatchDisambiguation enum."""
        try:
            return MatchDisambiguation[s.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown match disambiguation: {s}") from e


class _FormatPattern:
    """Base class for pattern matching and processing."""

    def __init__(self, pattern_def: dict[str, Any]):
        self._format_type = AnswerFormat.from_string(pattern_def["format_type"])
        self._match_disambiguation = MatchDisambiguation.from_string(
            pattern_def.get("match_disambiguation", "match_if_singular"),
        )
        self._pattern = re.compile(pattern_def.get("pattern", ""))
        self._num_captures = self._pattern.groups if self._pattern.groups > 0 else 1

        ct = pattern_def.get("capture_transform", {})
        self._transform_params = ct.get("params", [])
        self._transform_expr = ct.get("expr", None)
        if self._transform_expr is not None:
            assert (
                len(self._transform_params) == self._num_captures
            ), f"Capture transform must have {self._num_captures} parameters for pattern '{self._pattern.pattern}'"
        else:
            assert (
                len(self._transform_params) == 0
            ), f"Capture transform args given without an `expr` for pattern '{self._pattern.pattern}'"
            assert (
                self._num_captures == 1
            ), f"A capture transform must be provided for pattern '{self._pattern.pattern}' with {self._num_captures} arguments"

    @property
    def answer_format(self) -> AnswerFormat:
        """Return the answer format for this pattern."""
        return self._format_type

    def _capture_transform(self, *captures: Any) -> Any:
        """Transform the captured groups using the defined transformation function."""
        assert (
            len(captures) == self._num_captures
        ), f"Expected {self._num_captures} captures, but got {len(captures)}."
        if self._transform_expr is None:
            return captures[0]
        return evaluate_expr(
            self._transform_expr,
            names=dict(zip(self._transform_params, captures)),
        )

    def _match(self, s: str) -> list[str] | list[tuple[str, ...]]:
        """Match the string `s` with the regex pattern."""
        if self._match_disambiguation == MatchDisambiguation.MATCH_ALL:
            m = self._pattern.fullmatch(s)
            return [m.groups() if len(m.groups()) > 0 else (m.group(0),)] if m else []
        return self._pattern.findall(s)

    def __call__(self, s: str) -> tuple[Any, AnswerFormat] | None:
        """Match the string `s` with the regex pattern."""
        matches: list[str] | list[tuple[str, ...]] = self._match(s)
        if not matches:
            return None

        match_tuples: list[tuple[str, ...]] | None = None
        if all(isinstance(match, str) for match in matches):
            match_tuples = [(str(match_str),) for match_str in matches]
        elif all(isinstance(match, tuple) for match in matches):
            match_tuples = cast(list[tuple[str, ...]], matches)
        assert (
            match_tuples is not None
        ), f"Pattern '{self._pattern.pattern}' yielded mixed match types: {matches}"

        assert all(
            isinstance(match, tuple) and len(match) == self._num_captures
            for match in match_tuples
        ), f"Pattern '{self._pattern.pattern}' should yield {self._num_captures} captures per match, but got {matches}."

        if (
            len(match_tuples) == 1
            or self._match_disambiguation == MatchDisambiguation.MATCH_FIRST
        ):
            return self._capture_transform(*(match_tuples[0])), self.answer_format

        if self._match_disambiguation == MatchDisambiguation.MATCH_LAST:
            return self._capture_transform(*(match_tuples[-1])), self.answer_format

        if self._match_disambiguation == MatchDisambiguation.MATCH_IF_UNIQUE:
            # Note: Here we convert the captures to tuples to ensure lists are hashable.
            unique = {tuple(self._capture_transform(*match)) for match in match_tuples}
            if len(unique) == 1:
                return self._capture_transform(*match_tuples[0]), self.answer_format

        return None


_MATCH_TYPE = TypeVar("_MATCH_TYPE")
_OTHER_TYPE = TypeVar("_OTHER_TYPE")


class Matcher(ABC, Generic[_MATCH_TYPE]):
    """Base class for a matcher that matches a string."""

    @abstractmethod
    def __call__(self, s: str) -> _MATCH_TYPE:
        """Match the string `s`.

        Args:
            s: The string to match.

        Returns:
            A tuple of the matched value and its answer format. If no match is found,
            returns (None, AnswerFormat.INVALID).
        """


class _MatcherCompose(Matcher, Generic[_MATCH_TYPE]):
    """A matcher that composes two matchers into a new matcher."""

    def __init__(self, first: Matcher[str], second: Matcher[_MATCH_TYPE]):
        """Initialize with two matchers."""
        self.first = first
        self.second = second

    def __call__(self, s: str) -> _MATCH_TYPE:
        """Apply the matchers in sequence."""
        return self.second(self.first(s))


class SimpleMatcher(Matcher[str]):
    """Reduce a string to a subselection using a regex pattern."""

    def __init__(self, pattern_def: dict[str, Any]):
        """Initialize with a single pattern definition."""
        self._pattern = _FormatPattern(pattern_def)

    def __or__(self, other: "Matcher[_OTHER_TYPE]") -> "Matcher[_OTHER_TYPE]":
        """Allow composition of matchers with the | operator."""
        return _MatcherCompose(self, other)

    def __call__(self, s: str) -> str:
        """Match the string `s` with the regex pattern."""
        match = self._pattern(s)
        return cast(str, match[0]) if match is not None else ""


class SequentialMatcher(Matcher[tuple[Any, AnswerFormat]]):
    """Match a string against multiple patterns in sequence, returning the first match."""

    def __init__(self, *pattern_defs: dict[str, Any]):
        """Initialize with a list of pattern definitions to be matched in order."""
        self._patterns = [_FormatPattern(pattern_def) for pattern_def in pattern_defs]
        assert len(self._patterns) > 0, "At least one pattern must be provided."
        assert (
            self._patterns[0].answer_format == AnswerFormat.PROPER
        ), "The first pattern must have a proper answer format."
        assert all(
            pattern.answer_format != AnswerFormat.PROPER
            for pattern in self._patterns[1:]
        ), "Only the first pattern can have a proper answer format."

    def __call__(self, s: str) -> tuple[Any, AnswerFormat]:
        """Match the string `s` with the regex pattern."""
        for pattern in self._patterns:
            match = pattern(s)
            if match is not None:
                return match
        return None, AnswerFormat.INVALID
