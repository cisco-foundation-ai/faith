# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd

from faith._internal.algo.hash import dict_sha256
from faith._internal.algo.matching import AnswerFormat, SimpleMatcher
from faith._internal.algo.sampling import NShotSampler
from faith._internal.metrics.domain_specific_scores import LLMJudgeScore
from faith._internal.metrics.llm import llm_judge_grades, llm_metadata_metrics
from faith._internal.types.flags import GenerationMode
from faith.benchmark.benchmark import BaseBenchmark
from faith.benchmark.dataset.dataset import BenchmarkDataset
from faith.benchmark.formatting.qa import QAFormatter, QARecord
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader
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
            nshot_samlper=nshot_sampler,
            rng=rng,
            ancillary_columns=ancillary_columns,
        )

    def log_grader(
        self, model_format_config: dict[str, Any], recompute_stats: bool = False
    ) -> LogGrader:
        """Fetch a log grader for this benchmark."""
        op_cfg = self._config["output_processing"]
        if (
            self.generation_mode == GenerationMode.CHAT_COMPLETION
            and self._answer_type == LongAnswerType.FREE_FORM
        ):
            return JudgeBasedLogGrader(op_cfg, model_format_config, recompute_stats)
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
        nshot_samlper: NShotSampler,
        rng: np.random.Generator,
        ancillary_columns: frozenset[str] = frozenset(),
    ):
        """Initializes the LABenchmarkDataset with the given benchmark and parameters."""
        super().__init__(
            formatter,
            benchmark_data,
            nshot_samlper,
            rng,
            required_columns=frozenset({"question", "answer"}),
            ancillary_columns=ancillary_columns,
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
            subject=sample.get("subject", None),
            ancillary_data=self._extract_ancillary_data(sample),
        )


class JudgeBasedLogGrader(LogGrader):
    """A log grader for long-answer chat completions that are judged by an LLM."""

    def __init__(
        self,
        output_processing_config: dict[str, Any],
        model_format_config: dict[str, Any],
        recompute_stats: bool,
    ):
        """Initialize the judge-based log grader."""
        super().__init__(output_processing_config, model_format_config, recompute_stats)
        self._answer_extractor = SimpleMatcher(model_format_config)
        judge_config = output_processing_config.get("answer_judge", {})
        self._judge_score = LLMJudgeScore(judge_config)

    def _markup_entry_impl(self, log_entry: dict[str, Any]) -> dict[str, Any]:
        """Markup a single log entry with the computed statistics / scores."""
        correct_answer = cast(str, log_entry["data"]["label"])
        gen_answer: str | None = None
        answer_format = AnswerFormat.INVALID
        judgement: dict[str, Any] = {}
        # TODO(https://github.com/RobustIntelligence/faith/issues/286): Remove the use
        # of 'output_text' once we fully migrate to 'answer_text' at the next major
        # release.
        if (chat_comp := log_entry["model_data"].get("chat_comp", {})) and (
            answer_text := chat_comp.get("answer_text", None)
            or chat_comp.get("output_text", None)
        ):
            gen_answer, answer_format = (
                self._answer_extractor(answer_text),
                AnswerFormat.PROPER,
            )
            judgement = self._judge_score(correct_answer, gen_answer)

        log_entry["stats"] = {
            "label": correct_answer,
            "prediction": gen_answer,
            "answer_format": answer_format,
            "judgement": judgement,
            "subject": log_entry["data"].get("subject", None),
            "num_output_tokens": log_entry["model_data"]
            .get("chat_comp", {})
            .get("num_output_tokens", 0),
            "max_token_halt": log_entry["model_data"]
            .get("chat_comp", {})
            .get("max_token_halt", False),
        } | self._custom_scores(
            correct_answer,
            gen_answer,
            ancillary_data=log_entry["data"].get("ancillary_data", None),
        )
        return log_entry


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
        judgements: Sequence[dict[str, Any]] = kwargs.get("judgement", [])
        scores: Sequence[dict[str, float]] = kwargs.get("scores", [])
        num_output_tokens: Sequence[int] | None = kwargs.get("num_output_tokens", None)
        max_token_halt: Sequence[bool] | None = kwargs.get("max_token_halt", None)

        return (
            self._aggregate_scores(scores)  # Compute aggregate custom scores.
            | (
                llm_metadata_metrics(num_output_tokens, max_token_halt)
                if num_output_tokens is not None and max_token_halt is not None
                else {}
            )
            | llm_judge_grades(label, prediction, answer_format, judgements)
        )
