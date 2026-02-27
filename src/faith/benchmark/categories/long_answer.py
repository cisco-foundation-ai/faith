# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd

from faith._internal.algo.hash import dict_sha256
from faith._internal.algo.matching import AnswerFormat
from faith._internal.algo.sampling import NShotSampler
from faith._internal.metrics.llm import llm_basic_metrics, llm_metadata_metrics
from faith._internal.types.flags import GenerationMode
from faith.benchmark.benchmark import BaseBenchmark
from faith.benchmark.dataset.dataset import BenchmarkDataset
from faith.benchmark.formatting.qa import QAFormatter, QARecord
from faith.benchmark.grading.common_graders import ChatCompletionLogGrader
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader
from faith.benchmark.scores.types import Score
from faith.benchmark.types import BenchmarkSpec


class LongAnswerType(Enum):
    """Enum for validation types for long answer benchmarks."""

    # Long answer benchmarks where each answer is free-form text
    # to be evaluated by an LLM.
    FREE_FORM = "free_form"


class LABenchmark(BaseBenchmark):
    """Base `Benchmark` class for benchmarks with long answer question-answer pairs."""

    # Note: Because long answer benchmarks cannot be used in next-token or logits mode,
    # there is no current reason to implement an answer lead-in, which is
    # difficult to implement since long answer benchmarks do not have answer sets.

    def __init__(self, spec: BenchmarkSpec, config: dict[str, Any], **kwargs: Any):
        """Initializes the LABenchmark with the given specification and configuration."""
        super().__init__(spec=spec, config=config, **kwargs)
        assert spec.generation_mode not in [
            GenerationMode.LOGITS,
            GenerationMode.NEXT_TOKEN,
        ], "Long answer benchmarks do not support logits/next_token generation mode since long answers may be multiple tokens."
        self._answer_type = LongAnswerType(self._config["laqa_config"]["type"])
        assert (
            len(self._config.get("output_processing", {}).get("score_fns", {})) > 0
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
        model_format_config: dict[str, Any] | None = None,
        recompute_stats: bool = False,
    ) -> LogGrader:
        """Fetch a log grader for this benchmark."""
        op_cfg = self._config["output_processing"]
        if (
            self.generation_mode == GenerationMode.CHAT_COMPLETION
            and self._answer_type == LongAnswerType.FREE_FORM
        ):
            return ChatCompletionLogGrader(op_cfg, model_format_config, recompute_stats)
        raise ValueError(
            f"Unsupported generation mode: {self.generation_mode} for long answer log grading."
        )

    def grade_aggregator(self) -> GradeAggregator:
        """Fetch a grade aggregator for this benchmark."""
        return LAMetricsAggregator(self._config["output_processing"])


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
        self, index: int, sample: pd.Series, examples: Sequence[QARecord] | None = None
    ) -> QARecord:
        """Format a sample into a question-answer record."""
        return self._formatter.render_qa_record(
            index=index,
            sample_hash=dict_sha256(sample.to_dict()),
            raw_question=sample["question"],
            raw_answer=sample.get("answer", None),
            examples=examples,
            choice_map=None,  # Long answer benchmarks are not enumerable.
            subject=sample.get("subject", None),
            ancillary_data=self._extract_ancillary_data(sample),
        )


class LAMetricsAggregator(GradeAggregator):
    """The `GradeAggregator` for long answer benchmarks."""

    def _aggregate(self, **kwargs: Sequence[Any]) -> dict[str, Any]:
        """Computes the metrics for this benchmark for its collected sufficient statistics.

        Args:
            **kwargs: Keyword arguments containing metrics data.
                Expected keys: 'label', 'prediction', 'answer_format',
                    'scores', 'judgement'.

        Returns:
            Dictionary containing computed metrics.
        """
        label: Sequence[Any] = kwargs.get("label", [])
        prediction: Sequence[Any] = kwargs.get("prediction", [])
        answer_format: Sequence[AnswerFormat] = kwargs.get("answer_format", [])
        scores: Sequence[dict[str, Score]] = kwargs.get("scores", [])
        num_output_tokens: Sequence[int] | None = kwargs.get("num_output_tokens", None)
        max_token_halt: Sequence[bool] | None = kwargs.get("max_token_halt", None)

        return (
            self._aggregate_scores(scores)  # Compute aggregate custom scores.
            | (
                llm_metadata_metrics(num_output_tokens, max_token_halt)
                if num_output_tokens is not None and max_token_halt is not None
                else {}
            )
            | llm_basic_metrics(label, prediction, answer_format)
        )
