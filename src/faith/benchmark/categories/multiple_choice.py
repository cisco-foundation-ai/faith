# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""An implementation of multiple choice benchmarks.

This module provides the `MCBenchmark` class for multiple choice benchmarks, which
extends the `Benchmark` class to handle multiple choice question-answering tasks.
"""

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from transformers import PreTrainedTokenizerBase

from faith._internal.algo.hash import dict_sha256
from faith._internal.algo.matching import AnswerFormat
from faith._internal.algo.sampling import NShotSampler
from faith._internal.metrics.domain_specific_scores import Score
from faith._internal.metrics.llm import llm_metadata_metrics, llm_prediction_metrics
from faith._internal.metrics.types import SingleLabelSeq
from faith._internal.types.flags import GenerationMode
from faith.benchmark.benchmark import BaseBenchmark
from faith.benchmark.dataset.dataset import BenchmarkDataset
from faith.benchmark.formatting.qa import QAFormatter, QARecord
from faith.benchmark.grading.common_graders import (
    ChatCompletionLogGrader,
    LogitsLogGrader,
    NextTokenLogGrader,
)
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader
from faith.benchmark.types import BenchmarkSpec


def _load_answer_set(config: dict[str, Any]) -> frozenset[str]:
    """Get the space of all answer symbols from the benchmark's config.

    Args:
        config: The configuration dictionary for the benchmark.

    Returns:
        The set of answer symbols for the benchmark.
    """
    answer_symbols = config["mcqa_config"]["answer_symbols"]
    assert isinstance(
        answer_symbols, list
    ), f"Choices must be a list, but got {type(answer_symbols)}"

    answer_set = frozenset(answer_symbols)
    assert len(answer_set) == len(
        answer_symbols
    ), f"Choices must be unique, but got {answer_set}"
    assert len(answer_set) > 0, "Choices list is empty."
    assert all(
        isinstance(choice, str) for choice in answer_set
    ), f"Choices must be strings, but got {answer_set}"
    assert all(
        len(choice) == 1 for choice in answer_set
    ), f"Choices must be single characters, but got {answer_set}"
    assert all(
        choice.isalpha() and choice.isupper() for choice in answer_set
    ), f"Choices must be uppercase alphabetic characters, but got {answer_set}"

    return answer_set


class MCBenchmark(BaseBenchmark):
    """A benchmark for multiple choice question-answering tasks."""

    def __init__(self, spec: BenchmarkSpec, config: dict[str, Any], **kwargs: Any):
        """Initializes the multiple choice benchmark with the given specification."""
        super().__init__(spec=spec, config=config, **kwargs)

        self._answer_symbols = _load_answer_set(self._config)

    @property
    def answer_set(self) -> frozenset[str]:
        """Returns the space of answer symbols for a multiple choice benchmark."""
        return self._answer_symbols

    def answer_leadin(self, tokenizer: PreTrainedTokenizerBase) -> str:
        """Returns the answer lead-in for the benchmark."""
        answer_leadin_tokens, _, _ = self._split_answer_template(tokenizer)
        return tokenizer.decode(answer_leadin_tokens, skip_special_tokens=True)

    def answer_token_map(self, tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
        """Returns the mapping of answer symbols to their corresponding token id."""
        _, answer_map, _ = self._split_answer_template(tokenizer)
        return answer_map

    def _split_answer_template(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> tuple[list[int], dict[str, int], list[str]]:
        """Splits the answer template into its constituent tokenized components.

        For a given set of answers and tokenizer, this method assumses that all
        tokenizations of the rendered answers differ at exactly one
        identical token index. Given this, it returns a tuple of:
            - The list of tokens that precede the differing token index.
            - A mapping from each answer in `answer_set` to its unique token.
            - The list of tokens that follow the differing token index.
        To use this method, the `answer_set` must have at least two elements and they
        must each be tokenized as a single token within the template.
        """
        assert len(self.answer_set) > 1, "There must be at least two answers to split."
        answers = {
            symbol: self.formatter.render_answer(symbol) for symbol in self.answer_set
        }
        tokenizations = {
            symbol: tokenizer.encode(ans, add_special_tokens=False)
            for symbol, ans in answers.items()
        }
        assert (
            len(set(map(len, tokenizations.values()))) == 1
        ), "Different answers are encoded with different numbers of tokens."
        differences = [
            [ind for ind, val in enumerate(lst) if other[ind] != val]
            for k1, lst in tokenizations.items()
            for k2, other in tokenizations.items()
            if k1 < k2
        ]
        assert all(
            len(lst) == 1 for lst in differences
        ), "All pairs of answers must differ in exactly 1 token."
        differing_indices = list(set(lst[0] for lst in differences))
        assert len(differing_indices) == 1, "Answers differ at variable token indices."
        differing_index = differing_indices[0]

        a_tokenization = next(iter(tokenizations.values()))
        prefix_tokens = a_tokenization[:differing_index]
        suffix_tokens = a_tokenization[(differing_index + 1) :]
        answer_map = {
            symbol: tokenizations[symbol][differing_index]
            for symbol, tokens in tokenizations.items()
        }
        return (prefix_tokens, answer_map, suffix_tokens)

    def _build_dataset(
        self,
        benchmark_data: pd.DataFrame,
        nshot_sampler: NShotSampler,
        rng: np.random.Generator,
        randomize_choices: bool = False,
        ancillary_columns: frozenset[str] = frozenset(),
    ) -> BenchmarkDataset:
        """Builds the dataset for this benchmark."""
        return MCBenchmarkDataset(
            self.formatter,
            self.answer_set,
            benchmark_data,
            nshot_sampler,
            rng,
            randomize_choices,
            ancillary_columns=ancillary_columns,
        )

    def log_grader(
        self, model_format_config: dict[str, Any], recompute_stats: bool = False
    ) -> LogGrader:
        """Fetch a log grader for this benchmark."""
        op_cfg = self._config["output_processing"]
        if self.generation_mode == GenerationMode.LOGITS:
            return LogitsLogGrader(op_cfg, model_format_config, recompute_stats)
        if self.generation_mode == GenerationMode.NEXT_TOKEN:
            return NextTokenLogGrader(
                op_cfg, model_format_config, recompute_stats, self.answer_set
            )
        if self.generation_mode == GenerationMode.CHAT_COMPLETION:
            return ChatCompletionLogGrader(op_cfg, model_format_config, recompute_stats)
        raise ValueError(
            f"Unsupported generation mode: {self.generation_mode} for multiple-choice log grading."
        )

    def grade_aggregator(self) -> GradeAggregator:
        """Fetch a grade aggregator for this benchmark."""
        return MCMetricsAggregator(self._config["output_processing"], self.answer_set)


class MCBenchmarkDataset(BenchmarkDataset):
    """A dataset for multiple choice benchmarks.

    This dataset maps data loaded as a pandas DataFrame with columns:
        * 'question': the question text
        * 'choices': a list of answer choices ordered by the answer symbols
        * 'answer': the correct answer symbol (e.g., 'A', 'B', 'C', etc.)
        * 'subject' (optional): the subject of the question, if applicable
    into multiple choice examples that can be used for evaluation.
    """

    def __init__(
        self,
        formatter: QAFormatter,
        answer_set: frozenset[str],
        benchmark_data: pd.DataFrame,
        nshot_sampler: NShotSampler,
        rng: np.random.Generator,
        randomize_choices: bool = False,
        ancillary_columns: frozenset[str] = frozenset(),
    ):
        """Initializes the multiple choice benchmark dataset."""
        super().__init__(
            formatter,
            benchmark_data,
            nshot_sampler,
            rng,
            required_columns=frozenset({"question", "choices", "answer"}),
            ancillary_columns=ancillary_columns,
            optional_columns=frozenset({"subject"}),
        )
        self._answer_list = sorted(list(answer_set))
        self._randomize_choices = randomize_choices
        self._answer_permutation = list(range(len(self._answer_list)))
        if self._randomize_choices:
            self._rng.shuffle(self._answer_permutation)

    def _map_choices(self, sample: pd.Series) -> tuple[dict[str, str], str]:
        """Maps each answer symbol to its choice from the sample."""
        choices = list(sample["choices"])
        answer = str(sample["answer"])

        assert len(choices) == len(
            self._answer_list
        ), f"The choices {choices} cannot be mapped to {self._answer_list}"
        assert (
            answer in self._answer_list
        ), f"The answer '{answer}' is not in the answer set {self._answer_list}."
        annotated = [
            (choice, choice_letter == answer)
            for choice_letter, choice in zip(self._answer_list, choices)
        ]
        permuted = [annotated[i] for i in self._answer_permutation]
        remapped = dict(zip(self._answer_list, permuted))
        selected_answer = [
            choice_letter for choice_letter, choice in remapped.items() if choice[1]
        ]
        assert (
            len(selected_answer) == 1
        ), f"Expected exactly one correct answer, but got {len(selected_answer)}."
        permuted_symbol = selected_answer[0]
        permuted_choices = {k: v[0] for k, v in remapped.items()}
        assert (
            permuted_choices[permuted_symbol]
            == dict(zip(self._answer_list, choices))[answer]
        ), "Permuted choices do not match the original choices."
        return permuted_choices, permuted_symbol

    def _format_qa(
        self, index: int, sample: pd.Series, examples: Sequence[QARecord] | None = None
    ) -> QARecord:
        """Format a sample into a question-answer pair."""
        choice_map, correct_symbol = self._map_choices(sample)
        return self._formatter.render_qa_record(
            index=index,
            sample_hash=dict_sha256(sample.to_dict()),
            raw_question=sample["question"],
            raw_answer=correct_symbol,
            examples=examples,
            choice_map=choice_map,
            subject=sample.get("subject", None),
            ancillary_data=self._extract_ancillary_data(sample),
        )


class MCMetricsAggregator(GradeAggregator):
    """The `GradeAggregator` for multiple choice benchmarks."""

    def __init__(
        self,
        output_processing_config: dict[str, Any],
        answer_set: frozenset[str],
    ):
        """Initialize the metrics aggregator for multiple choice benchmarks."""
        super().__init__(output_processing_config)
        self._answer_list = sorted(list(answer_set))

    def _aggregate(self, **kwargs: Sequence[Any]) -> dict[str, Any]:
        """Computes the metrics for this benchmark for its collected sufficient statistics.

        Args:
            **kwargs: Keyword arguments containing metrics data.
                Expected keys: 'label', 'prediction', 'answer_format'.

        Returns:
            Dictionary containing computed metrics.
        """
        label: SingleLabelSeq = kwargs.get("label", [])
        prediction: SingleLabelSeq = kwargs.get("prediction", [])
        answer_format: Sequence[AnswerFormat] = kwargs.get("answer_format", [])
        scores: Sequence[dict[str, Score]] = kwargs.get("scores", [])
        subject: SingleLabelSeq | None = kwargs.get("subject", None)
        num_output_tokens: Sequence[int] | None = kwargs.get("num_output_tokens", None)
        max_token_halt: Sequence[bool] | None = kwargs.get("max_token_halt", None)

        stringified_preds = [p if p is not None else "" for p in prediction]
        extended_answers = self._answer_list + [""]
        return (
            self._aggregate_scores(scores)  # Compute aggregate custom scores.
            | (
                llm_metadata_metrics(num_output_tokens, max_token_halt)
                if num_output_tokens is not None and max_token_halt is not None
                else {}
            )
            | llm_prediction_metrics(
                label, prediction, answer_format, subject, frozenset(self._answer_list)
            )
            | {
                "f1_scores": dict(
                    zip(
                        self._answer_list,
                        f1_score(
                            label,
                            stringified_preds,
                            labels=self._answer_list,
                            average=None,
                            zero_division=np.nan,
                        ),
                    )
                ),
                "weighted_avg_f1": f1_score(
                    label,
                    stringified_preds,
                    labels=self._answer_list,
                    average="weighted",
                    zero_division=np.nan,
                ),
                "confusion_matrix_count": {
                    true_label: dict(zip(extended_answers, row))
                    for true_label, row in zip(
                        extended_answers,
                        confusion_matrix(
                            label, stringified_preds, labels=extended_answers
                        ).tolist(),
                    )
                },
            }
        )
