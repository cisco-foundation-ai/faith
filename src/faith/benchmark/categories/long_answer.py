# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from faith._internal.algo.hash import dict_sha256
from faith._internal.algo.sampling import NShotSampler
from faith._internal.metrics.llm import llm_basic_metrics, llm_metadata_metrics
from faith._internal.types.stats import MetricSummary
from faith._types.configs.benchmark import BenchmarkConfig, LongAnswerType
from faith._types.configs.patterns import AnswerFormat, PatternDef
from faith._types.records.prompt_record import PromptRecord
from faith.benchmark.benchmark import BaseBenchmark
from faith.benchmark.dataset.dataset import BenchmarkDataset
from faith.benchmark.formatting.qa import QAFormatter
from faith.benchmark.grading.common_graders import ChatCompletionLogGrader
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader
from faith.benchmark.scores.scoring import Score
from faith.benchmark.types import BenchmarkSpec
from faith.model.params import GenerationMode


class LABenchmark(BaseBenchmark):
    """Base `Benchmark` class for benchmarks with long answer question-answer pairs."""

    # Note: Because long answer benchmarks cannot be used in next-token or logits mode,
    # there is no current reason to implement an answer lead-in, which is
    # difficult to implement since long answer benchmarks do not have answer sets.

    def __init__(self, spec: BenchmarkSpec, config: BenchmarkConfig, **kwargs: Any):
        """Initializes the LABenchmark with the given specification and configuration."""
        super().__init__(spec, config, **kwargs)
        assert spec.generation_mode not in [
            GenerationMode.LOGITS,
            GenerationMode.NEXT_TOKEN,
        ], "Long answer benchmarks do not support logits/next_token generation mode since long answers may be multiple tokens."
        assert (
            self._config.laqa_config is not None
        ), "LAQAConfig is required for long answer benchmarks."
        self._answer_type = self._config.laqa_config.type
        assert (
            len(self._config.output_processing.score_fns) > 0
        ), "Long answer benchmarks must have at least one score function defined."

    def _build_dataset(
        self,
        benchmark_data: pd.DataFrame,
        nshot_sampler: NShotSampler,
        rng: np.random.Generator,
        randomize_choices: bool = False,  # noqa: ARG002
        ancillary_columns: frozenset[str] = frozenset(),
    ) -> BenchmarkDataset:
        """Builds the dataset for this benchmark."""
        return LABenchmarkDataset(
            formatter=self.formatter,
            benchmark_data=benchmark_data,
            nshot_sampler=nshot_sampler,
            rng=rng,
            ancillary_columns=ancillary_columns,
        )

    def log_grader(
        self,
        *,
        model_format_config: PatternDef | None = None,
        recompute_stats: bool = False,
    ) -> LogGrader:
        """Fetch a log grader for this benchmark."""
        if (
            self.generation_mode == GenerationMode.CHAT_COMP
            and self._answer_type == LongAnswerType.FREE_FORM
        ):
            return ChatCompletionLogGrader(
                self._config.output_processing,
                model_format_config,
                recompute_stats,
            )
        raise ValueError(
            f"Unsupported generation mode: {self.generation_mode} for long answer log grading."
        )

    def grade_aggregator(self) -> GradeAggregator:
        """Fetch a grade aggregator for this benchmark."""
        return LAMetricsAggregator(self._config.output_processing)


class LABenchmarkDataset(BenchmarkDataset):
    """Static dataset for long answer benchmarks."""

    def __init__(
        self,
        formatter: QAFormatter,
        benchmark_data: pd.DataFrame,
        nshot_sampler: NShotSampler,
        rng: np.random.Generator,
        ancillary_columns: frozenset[str] = frozenset(),
    ):
        """Initializes the LABenchmarkDataset with the given benchmark and parameters."""
        super().__init__(
            formatter,
            benchmark_data,
            nshot_sampler,
            rng,
            required_columns=frozenset({"question"}),
            ancillary_columns=ancillary_columns,
            optional_columns=frozenset({"answer", "subject"}),
        )

    def _format_qa(
        self,
        index: int,
        sample: pd.Series,
        examples: Sequence[PromptRecord] | None = None,
    ) -> PromptRecord:
        """Format a sample into a question-answer record."""
        return self._formatter.render_qa_record(
            index=index,
            sample_hash=dict_sha256(sample.to_dict()),
            raw_question=sample["question"],
            raw_answer=sample.get("answer"),
            examples=examples,
            choice_map=None,  # Long answer benchmarks are not enumerable.
            subject=sample.get("subject"),
            ancillary_data=self._extract_ancillary_data(sample),
        )


class LAMetricsAggregator(GradeAggregator):
    """The `GradeAggregator` for long answer benchmarks."""

    def _aggregate(self, **kwargs: Sequence[Any]) -> MetricSummary:
        """Computes the metrics for this benchmark for its collected sufficient statistics.

        Args:
            **kwargs: Keyword arguments containing metrics data.
                Expected keys: 'label', 'prediction', 'answer_format',
                    'scores', 'judgement'.

        Returns:
            Dictionary containing computed metrics.
        """
        label: Sequence[Any] = kwargs.get("label") or []
        prediction: Sequence[Any] = kwargs.get("prediction") or []
        answer_format: Sequence[AnswerFormat] = kwargs.get("answer_format") or []
        scores: Sequence[dict[str, Score]] = kwargs.get("scores") or []
        num_output_tokens: Sequence[int] | None = kwargs.get("num_output_tokens")
        max_token_halt: Sequence[bool] | None = kwargs.get("max_token_halt")

        return (
            self._aggregate_scores(scores)  # Compute aggregate custom scores.
            | (
                llm_metadata_metrics(num_output_tokens, max_token_halt)
                if num_output_tokens is not None and max_token_halt is not None
                else {}
            )
            | llm_basic_metrics(label, prediction, answer_format)
        )
