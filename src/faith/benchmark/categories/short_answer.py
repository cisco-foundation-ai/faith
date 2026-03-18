# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""An implementation of short answer benchmarks for evaluating LLMs.

This module provides the `SABenchmark` class for short answer benchmarks, which
extends the `Benchmark` class to handle short answer question-answering tasks.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from faith._internal.algo.hash import dict_sha256
from faith._internal.algo.sampling import NShotSampler
from faith._internal.metrics.llm import (
    llm_basic_metrics,
    llm_metadata_metrics,
    llm_multilabel_metrics,
    llm_prediction_metrics,
)
from faith._internal.types.stats import MetricSummary
from faith._types.config.benchmark import BenchmarkConfig, ShortAnswerType
from faith._types.config.patterns import AnswerFormat, PatternDef
from faith._types.config.scoring import OutputProcessingConfig
from faith._types.records.prompt_record import PromptRecord
from faith._types.records.stats import SingleLabelSeq
from faith.benchmark.benchmark import BaseBenchmark
from faith.benchmark.dataset.dataset import BenchmarkDataset
from faith.benchmark.formatting.qa import QAFormatter
from faith.benchmark.grading.common_graders import ChatCompletionLogGrader
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader
from faith.benchmark.scores.scoring import Score
from faith.benchmark.types import BenchmarkSpec
from faith.model.params import GenerationMode


class SABenchmark(BaseBenchmark):
    """Base `Benchmark` class for benchmarks with short answer question-answer pairs."""

    # Note: Because short answer benchmarks cannot be used in next-token or logits mode,
    # there is no current reason to implement an answer lead-in, which is
    # difficult to implement since short answer benchmarks do not have answer sets.

    def __init__(self, spec: BenchmarkSpec, config: BenchmarkConfig, **kwargs: Any):
        """Initializes the SABenchmark with the given specification and configuration."""
        super().__init__(spec, config, **kwargs)
        assert spec.generation_mode not in [
            GenerationMode.LOGITS,
            GenerationMode.NEXT_TOKEN,
        ], "Short answer benchmarks do not support logits/next_token generation mode since short answers may be multiple tokens."
        assert (
            self._config.saqa_config is not None
        ), "SAQAConfig is required for short answer benchmarks."
        self._answer_type = self._config.saqa_config.type
        if self._answer_type == ShortAnswerType.DOMAIN_SPECIFIC:
            assert (
                len(self._config.output_processing.score_fns) > 0
            ), "Domain-specific short answer benchmarks must have at least one score function defined."

    def _build_dataset(
        self,
        benchmark_data: pd.DataFrame,
        nshot_sampler: NShotSampler,
        rng: np.random.Generator,
        randomize_choices: bool = False,  # noqa: ARG002
        ancillary_columns: frozenset[str] = frozenset(),
    ) -> BenchmarkDataset:
        """Builds the dataset for this benchmark."""
        return SABenchmarkDataset(
            self.formatter,
            benchmark_data,
            nshot_sampler,
            rng,
            ancillary_columns=ancillary_columns,
        )

    def log_grader(
        self,
        *,
        model_format_config: PatternDef | None = None,
        recompute_stats: bool = False,
    ) -> LogGrader:
        """Fetch a log grader for this benchmark."""
        if self.generation_mode == GenerationMode.CHAT_COMP:
            return ChatCompletionLogGrader(
                self._config.output_processing, model_format_config, recompute_stats
            )
        raise ValueError(
            f"Unsupported generation mode: {self.generation_mode} for short answer log grading."
        )

    def grade_aggregator(self) -> GradeAggregator:
        """Fetch a grade aggregator for this benchmark."""
        return SAMetricsAggregator(self._config.output_processing, self._answer_type)


class SABenchmarkDataset(BenchmarkDataset):
    """Static dataset for short answer benchmarks."""

    def __init__(
        self,
        formatter: QAFormatter,
        benchmark_data: pd.DataFrame,
        nshot_sampler: NShotSampler,
        rng: np.random.Generator,
        ancillary_columns: frozenset[str] = frozenset(),
    ):
        """Initializes the SABenchmarkDataset with the given benchmark and parameters."""
        super().__init__(
            formatter,
            benchmark_data,
            nshot_sampler,
            rng,
            required_columns=frozenset({"question", "answer"}),
            ancillary_columns=ancillary_columns,
            optional_columns=frozenset({"subject"}),
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
            raw_answer=sample["answer"],
            examples=examples,
            choice_map=None,  # Short answer benchmarks are not enumerable.
            subject=sample.get("subject"),
            ancillary_data=self._extract_ancillary_data(sample),
        )


class SAMetricsAggregator(GradeAggregator):
    """The `GradeAggregator` for short answer benchmarks."""

    def __init__(
        self,
        output_processing_config: OutputProcessingConfig,
        answer_type: ShortAnswerType,
    ):
        """Initializes the SAMetricsAggregator with the given score function definitions."""
        super().__init__(output_processing_config)
        self._answer_type = answer_type

    def _aggregate(self, **kwargs: Sequence[Any]) -> MetricSummary:
        """Computes the metrics for this benchmark for its collected sufficient statistics.

        Args:
            **kwargs: Keyword arguments containing metrics data.
                Expected keys: 'label', 'prediction', 'answer_format', 'scores'.

        Returns:
            Dictionary containing computed metrics.
        """
        label: Sequence[Any] = kwargs.get("label") or []
        prediction: Sequence[Any] = kwargs.get("prediction") or []
        answer_format: Sequence[AnswerFormat] = kwargs.get("answer_format") or []
        scores: Sequence[dict[str, Score]] = kwargs.get("scores") or []
        subject: SingleLabelSeq | None = kwargs.get("subject")
        num_output_tokens: Sequence[int] | None = kwargs.get("num_output_tokens")
        max_token_halt: Sequence[bool] | None = kwargs.get("max_token_halt")

        agg_scores = self._aggregate_scores(scores)  # Compute aggregate custom scores.
        llm_metadata = (
            llm_metadata_metrics(num_output_tokens, max_token_halt)
            if num_output_tokens is not None and max_token_halt is not None
            else {}
        )

        if self._answer_type == ShortAnswerType.LABEL_SET:
            return (
                agg_scores
                | llm_metadata
                | llm_multilabel_metrics(label, prediction, answer_format)
            )
        if self._answer_type == ShortAnswerType.STRING_MATCH:
            return (
                agg_scores
                | llm_metadata
                | llm_prediction_metrics(
                    label, prediction, answer_format, subject, labelset=None
                )
            )
        if self._answer_type == ShortAnswerType.DOMAIN_SPECIFIC:
            return (
                agg_scores
                | llm_metadata
                | llm_basic_metrics(label, prediction, answer_format)
            )
        raise ValueError(f"Unsupported short answer type: {self._answer_type}.")
