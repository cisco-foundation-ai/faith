# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""An implementation of short answer benchmarks for evaluating LLMs.

This module provides the `SABenchmark` class for short answer benchmarks, which
extends the `Benchmark` class to handle short answer question-answering tasks.
"""
from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd

from faith._internal.algo.hash import dict_sha256
from faith._internal.algo.sampling import NShotSampler
from faith._internal.formatting import AnswerFormat
from faith._internal.metrics.llm import (
    llm_basic_metrics,
    llm_metadata_metrics,
    llm_multilabel_metrics,
    llm_prediction_metrics,
)
from faith._internal.metrics.types import SingleLabelSeq
from faith._internal.types.flags import GenerationMode
from faith.benchmark.benchmark import BaseBenchmark
from faith.benchmark.dataset.dataset import BenchmarkDataset
from faith.benchmark.formatting.qa import QAFormatter, QARecord
from faith.benchmark.grading.common_graders import ChatCompletionLogGrader
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader
from faith.benchmark.types import BenchmarkSpec


class ShortAnswerType(Enum):
    """Enum for validation types for short answer benchmarks."""

    # Short answer benchmarks where each answer is treated as a set of labels.
    LABEL_SET = "label_set"
    # Short answer benchmarks where each answer is treated as a single string prediction.
    STRING_MATCH = "string_match"
    # Short answer benchmarks where each answer is scored by domain-specific scores.
    DOMAIN_SPECIFIC = "domain_specific"


class SABenchmark(BaseBenchmark):
    """Base `Benchmark` class for benchmarks with short answer question-answer pairs."""

    # Note: Because short answer benchmarks cannot be used in next-token or logits mode,
    # there is no current reason to implement an answer lead-in, which is
    # difficult to implement since short answer benchmarks do not have answer sets.

    def __init__(self, spec: BenchmarkSpec, config: dict[str, Any], **kwargs: Any):
        """Initializes the SABenchmark with the given specification and configuration."""
        super().__init__(spec=spec, config=config, **kwargs)
        assert spec.generation_mode not in [
            GenerationMode.LOGITS,
            GenerationMode.NEXT_TOKEN,
        ], "Short answer benchmarks do not support logits/next_token generation mode since short answers may be multiple tokens."
        self._answer_type = ShortAnswerType(self._config["saqa_config"]["type"])
        if self._answer_type == ShortAnswerType.DOMAIN_SPECIFIC:
            assert (
                len(self._config.get("output_processing", {}).get("score_fns", {})) > 0
            ), "Domain-specific short answer benchmarks must have at least one score function defined."

    def _build_dataset(
        self,
        benchmark_data: pd.DataFrame,
        nshot_sampler: NShotSampler,
        rng: np.random.Generator,
        randomize_choices: bool = False,  # noqa: ARG002
    ) -> BenchmarkDataset:
        """Builds the dataset for this benchmark."""
        return SABenchmarkDataset(
            self.formatter,
            benchmark_data,
            nshot_sampler,
            rng,
        )

    def log_grader(self, recompute_stats: bool = False) -> LogGrader:
        """Fetch a log grader for this benchmark."""
        op_cfg = self._config["output_processing"]
        if self.generation_mode == GenerationMode.CHAT_COMPLETION:
            return ChatCompletionLogGrader(op_cfg, recompute_stats)
        raise ValueError(
            f"Unsupported generation mode: {self.generation_mode} for multiple-choice log grading."
        )

    def grade_aggregator(self) -> GradeAggregator:
        """Fetch a grade aggregator for this benchmark."""
        return SAMetricsAggregator(self._config["output_processing"], self._answer_type)


class SABenchmarkDataset(BenchmarkDataset):
    """Static dataset for short answer benchmarks."""

    def __init__(
        self,
        formatter: QAFormatter,
        benchmark_data: pd.DataFrame,
        nshot_samlper: NShotSampler,
        rng: np.random.Generator,
    ):
        """Initializes the SABenchmarkDataset with the given benchmark and parameters."""
        super().__init__(
            formatter,
            benchmark_data,
            nshot_samlper,
            rng,
            required_columns=frozenset({"question", "answer"}),
            optional_columns=frozenset({"subject"}),
        )

    def _format_qa(
        self, index: int, sample: pd.Series, examples: Sequence[QARecord] | None = None
    ) -> QARecord:
        """Format a sample into a question-answer record."""
        return self._formatter.render_qa_record(
            index=index,
            sample_hash=dict_sha256(sample.to_dict()),
            raw_question=sample["question"],
            raw_answer=sample["answer"],
            examples=examples,
            choice_map=None,  # Short answer benchmarks are not enumerable.
            subject=sample["subject"] if "subject" in sample else None,
        )


class SAMetricsAggregator(GradeAggregator):
    """The `GradeAggregator` for short answer benchmarks."""

    def __init__(
        self, output_processing_config: dict[str, Any], answer_type: ShortAnswerType
    ):
        """Initializes the SAMetricsAggregator with the given score function definitions."""
        super().__init__(output_processing_config)
        self._answer_type = answer_type

    def _aggregate(self, **kwargs: Sequence[Any]) -> dict[str, Any]:
        """Computes the metrics for this benchmark for its collected sufficient statistics.

        Args:
            **kwargs: Keyword arguments containing metrics data.
                Expected keys: 'label', 'prediction', 'answer_format', 'scores'.

        Returns:
            Dictionary containing computed metrics.
        """
        label: Sequence[Any] = kwargs.get("label", [])
        prediction: Sequence[Any] = kwargs.get("prediction", [])
        answer_format: Sequence[AnswerFormat] = kwargs.get("answer_format", [])
        scores: Sequence[dict[str, float]] = kwargs.get("scores", [])
        subject: SingleLabelSeq | None = kwargs.get("subject", None)
        num_output_tokens: Sequence[int] | None = kwargs.get("num_output_tokens", None)
        max_token_halt: Sequence[bool] | None = kwargs.get("max_token_halt", None)

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
