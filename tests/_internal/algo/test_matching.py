# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Sequence

import pytest

from faith._internal.algo.matching import (
    Match,
    SequentialMatcher,
    SimpleMatcher,
    _FormatPattern,
)
from faith._types.configs.patterns import (
    AnswerFormat,
    CaptureTransform,
    Disambiguation,
    PatternDef,
)


def test_format_pattern_init() -> None:
    pattern_def = PatternDef(
        format_type=AnswerFormat.PROPER,
        disambiguation=Disambiguation.MATCH_IF_SINGULAR,
        pattern=r"(\d+) (\d+)",
        capture_transform=CaptureTransform(params=["x", "y"], expr="int(x) + int(y)"),
    )
    pattern = _FormatPattern(pattern_def)
    # pylint: disable=protected-access
    assert pattern._format_type == AnswerFormat.PROPER
    assert pattern._disambiguation == Disambiguation.MATCH_IF_SINGULAR
    assert pattern._pattern.pattern == r"(\d+) (\d+)"
    assert pattern._num_captures == 2
    assert pattern._capture_transform(5, 7) == 12
    # pylint: enable=protected-access

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Capture transform must have 2 parameters for pattern '(\\d+) (\\d+)'"
        ),
    ):
        _FormatPattern(
            PatternDef(
                format_type=AnswerFormat.PROPER,
                disambiguation=Disambiguation.MATCH_IF_SINGULAR,
                pattern=r"(\d+) (\d+)",
                capture_transform=CaptureTransform(params=["x"], expr="int(x)"),
            )
        )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "A capture transform must be provided for pattern '(\\d+) (\\d+)' with 2 arguments"
        ),
    ):
        _FormatPattern(
            PatternDef(
                format_type=AnswerFormat.PROPER,
                disambiguation=Disambiguation.MATCH_IF_SINGULAR,
                pattern=r"(\d+) (\d+)",
            )
        )


def test_format_pattern_answer_format() -> None:
    assert (
        _FormatPattern(
            PatternDef(
                format_type=AnswerFormat.PROPER,
                disambiguation=Disambiguation.MATCH_IF_SINGULAR,
                pattern=r"(\d+) (\d+)",
                capture_transform=CaptureTransform(
                    params=["x", "y"], expr="int(x) + int(y)"
                ),
            )
        ).answer_format
        == AnswerFormat.PROPER
    )
    assert (
        _FormatPattern(
            PatternDef(
                format_type=AnswerFormat.IMPROPER,
                disambiguation=Disambiguation.MATCH_IF_UNIQUE,
                pattern=r"(\d+)",
            )
        ).answer_format
        == AnswerFormat.IMPROPER
    )
    assert (
        _FormatPattern(
            PatternDef(
                format_type=AnswerFormat.INFERRED,
                disambiguation=Disambiguation.MATCH_FIRST,
                pattern=r"(\d+)",
            )
        ).answer_format
        == AnswerFormat.INFERRED
    )
    assert (
        _FormatPattern(
            PatternDef(
                format_type=AnswerFormat.INVALID,
                disambiguation=Disambiguation.MATCH_LAST,
                pattern=r"(\d+)",
            )
        ).answer_format
        == AnswerFormat.INVALID
    )


@pytest.mark.parametrize(
    "spec, input_output_pairs",
    [
        (
            PatternDef(
                format_type=AnswerFormat.PROPER,
                disambiguation=Disambiguation.MATCH_IF_SINGULAR,
                pattern=r"(\d+) (\d+)",
                capture_transform=CaptureTransform(
                    params=["x", "y"], expr="int(x) + int(y)"
                ),
            ),
            [
                ("5 7", (12, AnswerFormat.PROPER)),
                ("5 7 or 8 2", None),
                ("abc", None),
            ],
        ),
        (
            PatternDef(
                format_type=AnswerFormat.IMPROPER,
                disambiguation=Disambiguation.MATCH_IF_UNIQUE,
                pattern=r"(\d+)",
            ),
            [
                ("57 abc", ("57", AnswerFormat.IMPROPER)),
                ("5 or 5 and 5", ("5", AnswerFormat.IMPROPER)),
                ("abc", None),
                ("55 7 8 2", None),
                ("abc", None),
            ],
        ),
        (
            PatternDef(
                format_type=AnswerFormat.INFERRED,
                disambiguation=Disambiguation.MATCH_FIRST,
                pattern=r"(\d+)",
            ),
            [
                ("5 abc", ("5", AnswerFormat.INFERRED)),
                ("5 or 6 and 7", ("5", AnswerFormat.INFERRED)),
                ("5 7 8 2", ("5", AnswerFormat.INFERRED)),
                ("abc", None),
            ],
        ),
        (
            PatternDef(
                format_type=AnswerFormat.INVALID,
                disambiguation=Disambiguation.MATCH_LAST,
                pattern=r"(\d+)",
            ),
            [
                ("5 abc", ("5", AnswerFormat.INVALID)),
                ("5 or 6 and 7", ("7", AnswerFormat.INVALID)),
                ("5 7 8 2", ("2", AnswerFormat.INVALID)),
                ("abc", None),
            ],
        ),
        (
            PatternDef(
                format_type=AnswerFormat.IMPROPER,
                disambiguation=Disambiguation.MATCH_ALL,
                pattern=r"(\d+)F",
            ),
            [
                ("448F", ("448", AnswerFormat.IMPROPER)),
                ("57F abc", None),
                ("abc", None),
                ("55F 7F", None),
                ("abc", None),
            ],
        ),
        (
            PatternDef(
                format_type=AnswerFormat.PROPER,
                disambiguation=Disambiguation.MATCH_ALL,
                pattern=r"(?s)\d+kg\s+.*",
            ),
            [
                ("945kg\n\nabc", ("945kg\n\nabc", AnswerFormat.PROPER)),
                ("weight: 57kg abc", None),
                ("abc", None),
                ("55kg,7kg", None),
                ("abc", None),
            ],
        ),
        (
            PatternDef(
                format_type=AnswerFormat.PROPER,
                disambiguation=Disambiguation.MATCH_ALL,
                pattern=r"(?is)\s*(\{.*\})\s*",
                capture_transform=CaptureTransform(
                    params=["dict_str"],
                    expr="from_json(dict_str).get('height')",
                ),
            ),
            [
                ('{\n  "height": 180,\n  "weight": 75\n}', (180, AnswerFormat.PROPER)),
                ('{\n  "weight": 75\n}', (None, AnswerFormat.PROPER)),
                ("Not a JSON string", None),
            ],
        ),
    ],
)
def test_format_pattern_call(
    spec: PatternDef,
    input_output_pairs: Sequence[tuple[str, tuple[str, AnswerFormat] | None]],
) -> None:
    """Test the call method of _FormatPattern."""
    pattern = _FormatPattern(spec)
    for input_str, expected_output in input_output_pairs:
        assert pattern(input_str) == expected_output


def test_simple_matcher() -> None:
    """Test the SimpleMatcher class."""
    matcher = SimpleMatcher(
        PatternDef(
            format_type=AnswerFormat.PROPER,
            disambiguation=Disambiguation.MATCH_IF_SINGULAR,
            pattern=r"(\d+)\s*\+\s*(?:\d+)",
        )
    )

    assert matcher("5 + 7") == Match("5", AnswerFormat.PROPER)
    assert matcher("5 + 7 or 5+7") == Match(None, AnswerFormat.INVALID)
    assert matcher("abc") == Match(None, AnswerFormat.INVALID)
    assert matcher("57 abc") == Match(None, AnswerFormat.INVALID)
    assert matcher("5 or 5 and 5") == Match(None, AnswerFormat.INVALID)


def test_sequential_matcher() -> None:
    """Test the SequentialMatcher class."""
    matcher = SequentialMatcher(
        PatternDef(
            format_type=AnswerFormat.PROPER,
            disambiguation=Disambiguation.MATCH_IF_SINGULAR,
            pattern=r"(\d+)\s*\+\s*(\d+)",
            capture_transform=CaptureTransform(
                params=["x", "y"], expr="str(int(x) + int(y))"
            ),
        ),
        PatternDef(
            format_type=AnswerFormat.IMPROPER,
            disambiguation=Disambiguation.MATCH_IF_UNIQUE,
            pattern=r"(\d+)",
        ),
    )

    assert matcher("5 + 7") == ("12", AnswerFormat.PROPER)
    assert matcher("5 + 7 or 5+7") == (None, AnswerFormat.INVALID)
    assert matcher("abc") == (None, AnswerFormat.INVALID)
    assert matcher("57 abc") == ("57", AnswerFormat.IMPROPER)
    assert matcher("5 or 5 and 5") == ("5", AnswerFormat.IMPROPER)


def test_sequential_matcher_assertions() -> None:
    with pytest.raises(AssertionError, match="At least one pattern must be provided."):
        SequentialMatcher()
    with pytest.raises(
        AssertionError, match="The first pattern must have a proper answer format."
    ):
        SequentialMatcher(
            PatternDef(
                format_type=AnswerFormat.IMPROPER,
                disambiguation=Disambiguation.MATCH_IF_UNIQUE,
                pattern=r"(\d+)",
            )
        )
    with pytest.raises(
        AssertionError, match="Only the first pattern can have a proper answer format."
    ):
        SequentialMatcher(
            PatternDef(
                format_type=AnswerFormat.PROPER,
                disambiguation=Disambiguation.MATCH_IF_SINGULAR,
                pattern=r"(\d+)",
            ),
            PatternDef(
                format_type=AnswerFormat.PROPER,
                disambiguation=Disambiguation.MATCH_IF_UNIQUE,
                pattern=r"(\w+)",
            ),
        )


def test_matcher_composition() -> None:
    """Test that matchers can be composed."""
    matcher1 = SimpleMatcher(
        PatternDef(
            format_type=AnswerFormat.PROPER,
            disambiguation=Disambiguation.MATCH_ALL,
            pattern=r"(\d+)\s*\+\s*(?:\d+)",
        )
    )
    matcher2 = SimpleMatcher(
        PatternDef(
            format_type=AnswerFormat.PROPER,
            disambiguation=Disambiguation.MATCH_ALL,
            pattern=r"(\d)(?:\d*)",
        )
    )
    matcher = matcher1 | matcher2

    assert matcher("85 + 7") == Match("8", AnswerFormat.PROPER)
    assert matcher("57 abc") == Match(None, AnswerFormat.INVALID)
    assert matcher("abc") == Match(None, AnswerFormat.INVALID)
    assert matcher("5 or 5 and 5") == Match(None, AnswerFormat.INVALID)
    assert matcher("5 + 7 or 5+7") == Match(None, AnswerFormat.INVALID)
