# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agent-trajectory log grader."""

from pathlib import Path

import pytest

from faith._internal.io.json import read_json_file
from faith._types.config.patterns import AnswerFormat
from faith._types.config.scoring import OutputProcessingConfig, ScoreFnConfig
from faith._types.json_types import JsonValue
from faith._types.record.model_response import GenerationError
from faith._types.record.sample import SampleRecord
from faith._types.record.score import Score
from faith._types.record.stats import Labeling, StatsRecord
from faith.agentic.core import AnswerExtractor
from faith.benchmark.grading.agentic_graders import AgentTrajectoryLogGrader
from tests.benchmark.qa.fake_record_maker import make_fake_record

_JACCARD = {"file_f1": ScoreFnConfig(type="jaccard")}
_TRAJECTORY = Path(__file__).parent / "testdata" / "agent-trajectory.json"


class _FileListExtractor(AnswerExtractor):
    """Extractor double: treats a list answer as a file list, anything else None.

    Also honors an abstain terminal tool: when it fired, the run means "no
    files" (empty list), disambiguated only by tool_name.
    """

    def extract(self, tool_name: str | None, answer: JsonValue) -> Labeling | None:
        if tool_name == "abstain":
            return []
        return [str(a) for a in answer] if isinstance(answer, list) else None


def _record(
    answer: JsonValue,
    label: JsonValue,
    answer_tool: str | None = "submit",
    stopped_reason: str = "submitted",
    error: str | None = None,
) -> SampleRecord:
    """Build a record carrying an agent trajectory with the given answer.

    `error` goes on the trajectory, which rejects it unless the reason is ERROR.
    """
    return make_fake_record(
        data={"label": label},
        model_data={
            "agent_trajectory": read_json_file(_TRAJECTORY)
            | {
                "answer": answer,
                "answer_tool": answer_tool,
                "stopped_reason": stopped_reason,
                "error": error,
            }
        },
    )


def _grader(
    score_fns: dict[str, ScoreFnConfig] | None = None,
) -> AgentTrajectoryLogGrader:
    return AgentTrajectoryLogGrader(
        OutputProcessingConfig(score_fns=score_fns or {}),
        extractor=_FileListExtractor(config={}),
    )


def test_empty_input_yields_no_records() -> None:
    """Test the grader is an identity over an empty log stream."""
    assert [log.stats for log in [] >> _grader()] == []


@pytest.mark.parametrize(
    "answer,answer_tool,label,prediction,answer_format",
    [
        # A submitted answer the extractor understands.
        (["a.c", "b.c"], "submit", ["b.c", "c.c"], ["a.c", "b.c"], AnswerFormat.PROPER),
        # Submitted, but not in a shape this benchmark can extract.
        ("just some prose", "submit", ["a.c"], None, AnswerFormat.IMPROPER),
        # No terminal tool ever fired: nothing was submitted at all.
        (None, None, ["a.c"], None, AnswerFormat.INVALID),
        # A terminal tool fired with a null answer. That IS a submission, so it
        # is IMPROPER rather than INVALID: answer_tool, not answer, decides.
        (None, "submit", ["a.c"], None, AnswerFormat.IMPROPER),
        # The extractor may branch on tool identity: the same null answer under
        # an abstain tool is a real, scorable "no files" prediction.
        (None, "abstain", [], [], AnswerFormat.PROPER),
    ],
    ids=["extracted", "unextractable", "never_submitted", "explicit_null", "abstained"],
)
def test_the_answer_format_records_whether_a_scorable_answer_arrived(
    answer: JsonValue,
    answer_tool: str | None,
    label: JsonValue,
    prediction: Labeling | None,
    answer_format: AnswerFormat,
) -> None:
    assert [
        log.stats
        for log in [_record(answer, label, answer_tool=answer_tool)] >> _grader()
    ] == [
        StatsRecord(
            label=label,
            prediction=prediction,
            answer_format=answer_format,
            stopped_reason="submitted",
        ),
    ]


@pytest.mark.parametrize(
    "stopped_reason",
    ["submitted", "no_tool_calls", "max_calls", "max_turns", "timeout"],
)
def test_why_the_loop_stopped_is_carried_into_the_stats(stopped_reason: str) -> None:
    """The stats are what reach the aggregator, so the reason must survive grading.

    Recorded for every reason, not just the ones that failed to answer: a rate is
    only readable against the total, and the answer-bearing case is the total.
    """
    stats = [
        log.stats
        for log in [_record(["a.c"], ["a.c"], stopped_reason=stopped_reason)]
        >> _grader()
    ]
    assert stats == [
        StatsRecord(
            label=["a.c"],
            prediction=["a.c"],
            answer_format=AnswerFormat.PROPER,
            stopped_reason=stopped_reason,
        ),
    ]


def test_missing_trajectory_is_invalid() -> None:
    """Test a record with no agent_trajectory at all is INVALID."""
    record = make_fake_record(data={"label": ["a.c"]})
    assert [log.stats for log in [record] >> _grader()] == [
        StatsRecord(
            label=["a.c"],
            prediction=None,
            answer_format=AnswerFormat.INVALID,
        ),
    ]


def test_scores_the_extracted_prediction_against_the_label() -> None:
    """The configured score_fns run on the extracted prediction.

    A real agentic benchmark must configure at least one score_fn, so this is
    the shape that actually ships; the other cases pin answer_format alone.
    """
    stats = [
        log.stats
        for log in [_record(["a.c", "b.c"], ["b.c", "c.c"])] >> _grader(_JACCARD)
    ]

    assert stats == [
        StatsRecord(
            label=["b.c", "c.c"],
            prediction=["a.c", "b.c"],
            answer_format=AnswerFormat.PROPER,
            # One of three files in the union matches: Jaccard 1/3.
            scores={"file_f1": Score(value=1 / 3, raw_value=1 / 3)},
            stopped_reason="submitted",
        ),
    ]


def test_an_unextractable_answer_still_scores_against_the_label() -> None:
    """An IMPROPER answer is scored as a miss, not skipped."""
    stats = [
        log.stats for log in [_record("just some prose", ["a.c"])] >> _grader(_JACCARD)
    ]

    assert stats == [
        StatsRecord(
            label=["a.c"],
            prediction=None,
            answer_format=AnswerFormat.IMPROPER,
            scores={"file_f1": Score(value=0.0, raw_value=0.0)},
            stopped_reason="submitted",
        ),
    ]


def test_a_run_error_is_carried_into_the_stats() -> None:
    """Test the record's generation error reaches stats, where it excludes the record."""
    record = _record(
        None, ["a.c"], answer_tool=None, stopped_reason="error", error="container died"
    )
    record.model_data.error = GenerationError(
        title="Agent Run Error", details="container died"
    )
    assert [log.stats for log in [record] >> _grader()] == [
        StatsRecord(
            label=["a.c"],
            prediction=None,
            answer_format=AnswerFormat.INVALID,
            error="Agent Run Error",
            stopped_reason="error",
        ),
    ]
