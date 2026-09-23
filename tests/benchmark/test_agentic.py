# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agentic benchmark type."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from faith._internal.iter.transform import Transform
from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.benchmark.spec import BenchmarkSpec
from faith._types.config.agentic import (
    AgenticAnswerExtractorConfig,
    AgenticConfig,
    AgenticEnvironmentConfig,
    AgenticTaskLoaderConfig,
)
from faith._types.config.benchmark import BenchmarkConfig
from faith._types.config.patterns import AnswerFormat
from faith._types.config.scoring import OutputProcessingConfig, ScoreFnConfig
from faith._types.json_types import JsonValue
from faith._types.model.generation import GenerationMode
from faith._types.model.prompt import PromptFormatter
from faith._types.record.model_response import GenerationError
from faith._types.record.sample import SampleRecord
from faith._types.record.stats import Labeling, MetricSummary, StatsRecord
from faith.agentic.core import AnswerExtractor
from faith.agentic.core.task import Task
from faith.benchmark.agentic import AgenticBenchmark
from tests.benchmark.qa.fake_record_maker import make_fake_record


class _FakeExtractor(AnswerExtractor):
    """Extractor double: a list answer becomes a file list, anything else None."""

    def extract(self, tool_name: str | None, answer: JsonValue) -> Labeling | None:
        return [str(a) for a in answer] if isinstance(answer, list) else None


def _spec(name: str = "fake-agentic") -> BenchmarkSpec:
    # Agentic pins 3 of 4 spec fields; only the name varies.
    return BenchmarkSpec(
        name=name,
        generation_mode=GenerationMode.CHAT_COMP,
        prompt_format=PromptFormatter.CHAT,
        n_shot=SampleRatio(0),
    )


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        output_processing=OutputProcessingConfig(
            primary_metric="mean_file_f1",
            score_fns={"file_f1": ScoreFnConfig(type="jaccard")},
        ),
        agentic_config=AgenticConfig(
            task_loader=AgenticTaskLoaderConfig(type="fake"),
            environment=AgenticEnvironmentConfig(type="fake"),
            answer_extractor=AgenticAnswerExtractorConfig(type="fake"),
        ),
    )


def _record(
    answer: JsonValue,
    label: JsonValue,
    stopped_reason: str = "submitted",
    error: str | None = None,
) -> SampleRecord:
    return make_fake_record(
        data={"label": label},
        model_data={
            "agent_trajectory": {
                "task_id": "t1",
                "answer": answer,
                "messages": [],
                "tool_calls": [],
                "tool_responses": [],
                "num_tool_calls": 0,
                "execution_time_seconds": 0.0,
                "stopped_reason": stopped_reason,
                "error": error,
            }
        },
    )


def _errored_record(label: JsonValue) -> SampleRecord:
    """A record whose run broke: excluded from scoring, but still a stopped reason."""
    record = _record(None, label, stopped_reason="error", error="container died")
    record.model_data.error = GenerationError(
        title="Agent Run Error", details="container died"
    )
    return record


def _benchmark(seed: int = 0) -> AgenticBenchmark:
    return AgenticBenchmark(_spec(), _config(), path=Path("/fake"), seed=seed)


def _log_grader() -> Transform:
    """The benchmark's grader, with the configured extractor resolved to a fake."""
    with patch(
        "faith.benchmark.agentic.build_answer_extractor",
        lambda cfg: _FakeExtractor(config={}),
    ):
        return _benchmark().log_grader()


def _load_tasks(tasks: list[Task], sample_size: int | None = None):
    """The benchmark's loaded tasks, with the configured loader resolved to a fake."""
    with patch(
        "faith.benchmark.agentic.build_task_loader",
        lambda cfg: SimpleNamespace(load_tasks=lambda path: list(tasks)),
    ):
        return _benchmark().load_tasks(sample_size=sample_size)


def _summarize(records: list[SampleRecord]) -> MetricSummary:
    """Grade then aggregate, as the eval phase does."""
    graded = list(records >> _log_grader())
    return _benchmark().grade_aggregator()([g.stats for g in graded])


def _tasks(count: int) -> list[Task]:
    return [Task(id=f"t{i}", prompt="p", ground_truth=None) for i in range(count)]


def test_requires_agentic_config() -> None:
    """An agentic benchmark without an agentic_config fails fast."""
    with pytest.raises(AssertionError, match="AgenticConfig is required"):
        AgenticBenchmark(_spec(), BenchmarkConfig())


def test_log_grader_scores_a_trajectory_with_the_configured_extractor() -> None:
    """log_grader resolves the configured extractor and scores against the label.

    Covers the wiring and the grading together: a grader built without the
    resolved extractor could not produce a prediction at all.
    """
    [graded] = list([_record(["a.c", "b.c"], ["b.c", "c.c"])] >> _log_grader())

    assert graded.stats == StatsRecord(
        label=["b.c", "c.c"],
        prediction=["a.c", "b.c"],
        answer_format=AnswerFormat.PROPER,
        subject=None,
        # Jaccard of {a.c, b.c} against {b.c, c.c}: |{b.c}| / |{a.c, b.c, c.c}|.
        scores={
            "file_f1": {
                "raw_value": pytest.approx(1 / 3),
                "value": pytest.approx(1 / 3),
            }
        },
        stopped_reason="submitted",
    )


def test_grade_aggregator_summarizes_scores_and_formats() -> None:
    """The aggregator reports the custom scores plus the answer-format breakdown.

    Compared whole: QA's accuracy metrics must NOT appear here, because agentic
    labels are arbitrary JSON, not the single strings those metrics assume.
    """
    assert _summarize(
        [
            _record(["a.c", "b.c"], ["a.c", "b.c"]),  # perfect overlap
            _record(None, ["a.c"]),  # never submitted -> INVALID
        ]
    ) == {
        "query_count": 2,
        "mean_file_f1": pytest.approx(0.5),
        "median_file_f1": pytest.approx(0.5),
        "format_count": {"proper": 1, "improper": 0, "inferred": 0, "invalid": 1},
        "format_rate": {
            "proper": pytest.approx(0.5),
            "improper": pytest.approx(0.0),
            "inferred": pytest.approx(0.0),
            "invalid": pytest.approx(0.5),
        },
        "scored_count": 2,
        "error_count": 0,
        "error_rate": pytest.approx(0.0),
        "stopped_reason_count": {
            "submitted": 2,
            "no_tool_calls": 0,
            "max_calls": 0,
            "max_turns": 0,
            "timeout": 0,
            "error": 0,
        },
        "stopped_reason_rate": {
            "submitted": pytest.approx(1.0),
            "no_tool_calls": pytest.approx(0.0),
            "max_calls": pytest.approx(0.0),
            "max_turns": pytest.approx(0.0),
            "timeout": pytest.approx(0.0),
            "error": pytest.approx(0.0),
        },
    }


def test_the_stopped_reason_breakdown_separates_runs_that_never_answered() -> None:
    """Two runs scoring 0.0 for different reasons are distinguishable in the metrics.

    This is the whole point of the breakdown: an agent nudged out without calling
    submit and an agent that answered wrongly are the same number otherwise.
    """
    summary = _summarize(
        [
            _record(["a.c"], ["a.c"]),
            _record(None, ["a.c"], stopped_reason="no_tool_calls"),
            _record(None, ["a.c"], stopped_reason="no_tool_calls"),
            _record(None, ["a.c"], stopped_reason="max_calls"),
        ]
    )
    assert summary["stopped_reason_count"] == {
        "submitted": 1,
        "no_tool_calls": 2,
        "max_calls": 1,
        "max_turns": 0,
        "timeout": 0,
        "error": 0,
    }
    assert summary["stopped_reason_rate"] == {
        "submitted": pytest.approx(0.25),
        "no_tool_calls": pytest.approx(0.5),
        "max_calls": pytest.approx(0.25),
        "max_turns": pytest.approx(0.0),
        "timeout": pytest.approx(0.0),
        "error": pytest.approx(0.0),
    }


def test_an_errored_run_is_counted_in_the_stopped_reason_breakdown() -> None:
    """An errored record is excluded from scoring but still counted as a reason.

    Otherwise a run where everything errored would show an empty-looking breakdown
    beside a nonzero error_count.
    """
    summary = _summarize([_record(["a.c"], ["a.c"]), _errored_record(["a.c"])])
    assert (summary["error_count"], summary["scored_count"]) == (1, 1)
    assert summary["stopped_reason_count"] == {
        "submitted": 1,
        "no_tool_calls": 0,
        "max_calls": 0,
        "max_turns": 0,
        "timeout": 0,
        "error": 1,
    }


def test_aggregating_zero_records_reports_counts_alone() -> None:
    """An empty trial reports its counts rather than a summary of nans."""
    assert _benchmark().grade_aggregator()([]) == {
        "query_count": 0,
        "scored_count": 0,
        "error_count": 0,
        "error_rate": pytest.approx(float("nan"), nan_ok=True),
    }


def test_load_tasks_returns_all_tasks_when_unsampled() -> None:
    """Without a sample size, every task is returned in order, paired with its index."""
    tasks = _tasks(5)
    assert _load_tasks(tasks) == list(enumerate(tasks))


@pytest.mark.parametrize("sample_size", [3, 5, 99])
def test_load_tasks_indexes_by_unsampled_position(sample_size: int) -> None:
    """A task keeps the same index at every sample size, so resume matches by index.

    This is the benchmark's own logic: tasks are enumerated *before* sampling.
    Numbering the sampled subset 0..n-1 instead would move a task's index when
    `--sample-size` changes, re-running trajectories the log already has. The
    draw's order-preservation, cross-size stability, and the `n >= len(items)`
    passthrough belong to `sample_sequence` (see
    `tests/_internal/algo/test_sampling.py`).
    """
    sampled = _load_tasks(_tasks(5), sample_size=sample_size)

    assert len(sampled) == min(sample_size, 5)
    assert [task.id for _, task in sampled] == [f"t{i}" for i, _ in sampled]
