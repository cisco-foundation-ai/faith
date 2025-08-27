# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Sequence

import pytest

from faith._internal.algo.matching import (
    AnswerFormat,
    MatchDisambiguation,
    SequentialMatcher,
    _FormatPattern,
)


def test_answer_format_str() -> None:
    assert str(AnswerFormat.PROPER) == "proper"
    assert str(AnswerFormat.IMPROPER) == "improper"
    assert str(AnswerFormat.INFERRED) == "inferred"
    assert str(AnswerFormat.INVALID) == "invalid"


def test_answer_format_from_string() -> None:
    assert AnswerFormat.from_string("proper") == AnswerFormat.PROPER
    assert AnswerFormat.from_string("improper") == AnswerFormat.IMPROPER
    assert AnswerFormat.from_string("inferred") == AnswerFormat.INFERRED
    assert AnswerFormat.from_string("invalid") == AnswerFormat.INVALID

    with pytest.raises(ValueError, match="Unknown answer format: unknown"):
        AnswerFormat.from_string("unknown")


def test_match_disambiguation_str() -> None:
    assert str(MatchDisambiguation.MATCH_IF_SINGULAR) == "match_if_singular"
    assert str(MatchDisambiguation.MATCH_IF_UNIQUE) == "match_if_unique"
    assert str(MatchDisambiguation.MATCH_FIRST) == "match_first"
    assert str(MatchDisambiguation.MATCH_LAST) == "match_last"


def test_match_disambiguation_from_string() -> None:
    assert (
        MatchDisambiguation.from_string("match_if_singular")
        == MatchDisambiguation.MATCH_IF_SINGULAR
    )
    assert (
        MatchDisambiguation.from_string("match_if_unique")
        == MatchDisambiguation.MATCH_IF_UNIQUE
    )
    assert (
        MatchDisambiguation.from_string("match_first")
        == MatchDisambiguation.MATCH_FIRST
    )
    assert (
        MatchDisambiguation.from_string("match_last") == MatchDisambiguation.MATCH_LAST
    )

    with pytest.raises(ValueError, match="Unknown match disambiguation: unknown"):
        MatchDisambiguation.from_string("unknown")


def test_format_pattern_init() -> None:
    pattern_def = {
        "format_type": "proper",
        "match_disambiguation": "match_if_singular",
        "pattern": r"(\d+) (\d+)",
        "capture_transform": {"params": ["x", "y"], "expr": "int(x) + int(y)"},
    }
    pattern = _FormatPattern(pattern_def)
    assert pattern._format_type == AnswerFormat.PROPER
    assert pattern._match_disambiguation == MatchDisambiguation.MATCH_IF_SINGULAR
    assert pattern._pattern.pattern == r"(\d+) (\d+)"
    assert pattern._num_captures == 2
    assert pattern._capture_transform(5, 7) == 12

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Capture transform must have 2 parameters for pattern '(\\d+) (\\d+)'"
        ),
    ):
        _FormatPattern(
            {
                "format_type": "proper",
                "match_disambiguation": "match_if_singular",
                "pattern": r"(\d+) (\d+)",
                "capture_transform": {"params": ["x"], "expr": "int(x)"},
            }
        )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "A capture transform must be provided for pattern '(\\d+) (\\d+)' with 2 arguments"
        ),
    ):
        _FormatPattern(
            {
                "format_type": "proper",
                "match_disambiguation": "match_if_singular",
                "pattern": r"(\d+) (\d+)",
            }
        )


def test_format_pattern_answer_format() -> None:
    assert (
        _FormatPattern(
            {
                "format_type": "proper",
                "match_disambiguation": "match_if_singular",
                "pattern": r"(\d+) (\d+)",
                "capture_transform": {"params": ["x", "y"], "expr": "int(x) + int(y)"},
            }
        ).answer_format
        == AnswerFormat.PROPER
    )
    assert (
        _FormatPattern(
            {
                "format_type": "improper",
                "match_disambiguation": "match_if_unique",
                "pattern": r"(\d+)",
            }
        ).answer_format
        == AnswerFormat.IMPROPER
    )
    assert (
        _FormatPattern(
            {
                "format_type": "inferred",
                "match_disambiguation": "match_first",
                "pattern": r"(\d+)",
            }
        ).answer_format
        == AnswerFormat.INFERRED
    )
    assert (
        _FormatPattern(
            {
                "format_type": "invalid",
                "match_disambiguation": "match_last",
                "pattern": r"(\d+)",
            }
        ).answer_format
        == AnswerFormat.INVALID
    )


@pytest.mark.parametrize(
    "spec, input_output_pairs",
    [
        (
            {
                "format_type": "proper",
                "match_disambiguation": "match_if_singular",
                "pattern": r"(\d+) (\d+)",
                "capture_transform": {"params": ["x", "y"], "expr": "int(x) + int(y)"},
            },
            [
                ("5 7", (12, AnswerFormat.PROPER)),
                ("5 7 or 8 2", None),
                ("abc", None),
            ],
        ),
        (
            {
                "format_type": "improper",
                "match_disambiguation": "match_if_unique",
                "pattern": r"(\d+)",
            },
            [
                ("57 abc", ("57", AnswerFormat.IMPROPER)),
                ("5 or 5 and 5", ("5", AnswerFormat.IMPROPER)),
                ("abc", None),
                ("55 7 8 2", None),
                ("abc", None),
            ],
        ),
        (
            {
                "format_type": "inferred",
                "match_disambiguation": "match_first",
                "pattern": r"(\d+)",
            },
            [
                ("5 abc", ("5", AnswerFormat.INFERRED)),
                ("5 or 6 and 7", ("5", AnswerFormat.INFERRED)),
                ("5 7 8 2", ("5", AnswerFormat.INFERRED)),
                ("abc", None),
            ],
        ),
        (
            {
                "format_type": "invalid",
                "match_disambiguation": "match_last",
                "pattern": r"(\d+)",
            },
            [
                ("5 abc", ("5", AnswerFormat.INVALID)),
                ("5 or 6 and 7", ("7", AnswerFormat.INVALID)),
                ("5 7 8 2", ("2", AnswerFormat.INVALID)),
                ("abc", None),
            ],
        ),
    ],
)
def test_format_pattern_call(
    spec: dict[str, str],
    input_output_pairs: Sequence[tuple[str, tuple[str, AnswerFormat] | None]],
) -> None:
    """Test the call method of _FormatPattern."""
    pattern = _FormatPattern(spec)
    for input_str, expected_output in input_output_pairs:
        assert pattern(input_str) == expected_output


def test_sequential_matcher() -> None:
    """Test the SequentialMatcher class."""
    matcher = SequentialMatcher(
        {
            "format_type": "proper",
            "match_disambiguation": "match_if_singular",
            "pattern": r"(\d+)\s*\+\s*(\d+)",
            "capture_transform": {"params": ["x", "y"], "expr": "str(int(x) + int(y))"},
        },
        {
            "format_type": "improper",
            "match_disambiguation": "match_if_unique",
            "pattern": r"(\d+)",
        },
    )

    assert matcher("5 + 7") == ("12", AnswerFormat.PROPER)
    assert matcher("5 + 7 or 5+7") == (None, AnswerFormat.INVALID)
    assert matcher("abc") == (None, AnswerFormat.INVALID)
    assert matcher("57 abc") == ("57", AnswerFormat.IMPROPER)
    assert matcher("5 or 5 and 5") == ("5", AnswerFormat.IMPROPER)


def test_sequential_matcher_assertions() -> None:
    with pytest.raises(AssertionError, match="At least one pattern must be provided."):
        _matcher_no_patterns = SequentialMatcher()
    with pytest.raises(
        AssertionError, match="The first pattern must have a proper answer format."
    ):
        _matcher_invalid_first = SequentialMatcher(
            {
                "format_type": "improper",
                "match_disambiguation": "match_if_unique",
                "pattern": r"(\d+)",
            }
        )
    with pytest.raises(
        AssertionError, match="Only the first pattern can have a proper answer format."
    ):
        _matcher_invalid_subsequent = SequentialMatcher(
            {
                "format_type": "proper",
                "match_disambiguation": "match_if_singular",
                "pattern": r"(\d+)",
            },
            {
                "format_type": "proper",
                "match_disambiguation": "match_if_unique",
                "pattern": r"(\w+)",
            },
        )
