# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""An agentic benchmark: scores an autonomous agent's trajectory over a set of tasks."""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from faith._internal.algo.sampling import sample_sequence
from faith._internal.metrics.llm import formatting_metrics
from faith._types.benchmark.spec import BenchmarkSpec
from faith._types.config.agentic import AgenticConfig
from faith._types.config.benchmark import BenchmarkConfig
from faith._types.config.patterns import AnswerFormat, PatternDef
from faith._types.record.score import Score
from faith._types.record.stats import MetricSummary
from faith.agentic.core.task import Task
from faith.agentic.orchestration import build_answer_extractor, build_task_loader
from faith.benchmark.benchmark import Benchmark
from faith.benchmark.grading.agentic_graders import AgentTrajectoryLogGrader
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader

logger = logging.getLogger(__name__)


class AgenticBenchmark(Benchmark):
    """A benchmark that scores an agent's trajectory over a set of tasks.

    Subclasses the neutral `Benchmark` base rather than `QABenchmark`: there is no
    dataset sampling, prompt formatting, n-shot holdout, or generation mode. What
    it shares with QA is grading — `log_grader` and `grade_aggregator` reuse QA's
    scoring machinery over the same `output_processing` config.

    Requires an `agentic_config` and at least one score function in
    `output_processing` (the metric applied to the extractor's prediction).
    Execution belongs to the TrialRunner; this type only describes and grades.
    """

    def __init__(self, spec: BenchmarkSpec, config: BenchmarkConfig, **kwargs: Any):
        """Initialize the agentic benchmark from its spec and config."""
        super().__init__(spec, config, **kwargs)
        # Narrowing for the type checker; load_benchmark only routes here when set.
        assert (
            self._config.agentic_config is not None
        ), "AgenticConfig is required for agentic benchmarks."

    @property
    def agentic_config(self) -> AgenticConfig:
        """Fetch the agentic configuration for this benchmark."""
        assert self._config.agentic_config is not None
        return self._config.agentic_config

    def load_tasks(self, sample_size: int | None = None) -> list[tuple[int, Task]]:
        """Load this benchmark's agent tasks, each paired with its dataset position.

        The agentic analog of `QABenchmark.build_dataset`. The position is the
        task's index in the unsampled list, like QA's preserved row label: the
        reconciliation key is built from it, so a resume at a different
        `sample_size` still matches the trajectories it already has.

        If `sample_size` is given, a seeded random subset of that many tasks is
        returned in their original order, matching QA's `--sample-size` semantics.
        """
        assert (
            self._path is not None
        ), "Agentic benchmark requires a data path to load tasks."
        tasks = list(
            enumerate(build_task_loader(self.agentic_config).load_tasks(self._path))
        )
        if sample_size is None:
            return tasks
        if self._seed is None:
            logger.warning("No seed provided for benchmark; using default seed.")
        return sample_sequence(tasks, sample_size, np.random.default_rng(self._seed))

    def log_grader(
        self,
        *,
        model_format_config: PatternDef | None = None,
    ) -> LogGrader:
        """Fetch a log grader that scores this benchmark's agent trajectories.

        The extractor named in the agentic config is resolved and injected into
        the grader, which turns each agent's submitted answer into a prediction
        and scores it against the ground-truth label with the configured metrics.
        """
        return AgentTrajectoryLogGrader(
            self._config.output_processing,
            extractor=build_answer_extractor(self.agentic_config.answer_extractor),
        )

    def grade_aggregator(self) -> GradeAggregator:
        """Fetch a grade aggregator that summarizes this benchmark's graded records."""
        return AgenticMetricsAggregator(self._config.output_processing)


class AgenticMetricsAggregator(GradeAggregator):
    """The `GradeAggregator` for agentic benchmarks.

    Aggregates the configured custom scores (the primary metric — e.g. file_f1 —
    lives here) and the answer-format breakdown (PROPER/IMPROPER/INVALID rates,
    the agentic signal for how often the agent produced a scorable answer). It
    does NOT compute QA's accuracy metrics: agentic labels and predictions are
    arbitrary JSON shapes, not the single strings those metrics assume.
    """

    def _aggregate(self, **kwargs: Sequence[Any]) -> MetricSummary:
        """Aggregate custom scores and answer-format rates from the collected stats."""
        answer_format: Sequence[AnswerFormat] = kwargs.get("answer_format") or []
        scores: Sequence[dict[str, Score]] | None = kwargs.get("scores")
        # Averaging nothing is nan, not zero, so report the count alone rather than
        # a summary of nans (and the numpy warnings that produce them).
        if not answer_format:
            return {"query_count": 0}
        return (
            {"query_count": len(answer_format)}
            | self._aggregate_scores(scores)
            | formatting_metrics(answer_format, None)
        )
