# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides common log graders for benchmarks."""

import re
from typing import Any

from faith._internal.algo.matching import (
    AllMatcher,
    Matcher,
    SequentialMatcher,
    SimpleMatcher,
)
from faith._types.config.patterns import AnswerFormat, PatternDef
from faith._types.config.scoring import OutputProcessingConfig
from faith._types.records.sample_record import SampleRecord
from faith._types.records.stats import Labeling, StatsRecord
from faith.benchmark.grading.log_grader import LogGrader


class LogitsLogGrader(LogGrader):
    """A log grader for multiple choice benchmarks that log the next-token logits."""

    def _markup_entry_impl(self, log_entry: SampleRecord) -> SampleRecord:
        """Markup a single log entry with the computed statistics / scores."""
        label: str | None = log_entry.data.label
        extracted_pred: str | None = None
        answer_format = AnswerFormat.INVALID
        log_probs: dict[str, float] | None = None
        if label is not None and (logits := log_entry.model_data.logits):
            # TODO(https://github.com/cisco-foundation-ai/faith/issues/26):
            # Handle multiple logits entries; currently assumes only one entry.
            first_token_logits = logits[0] if len(logits) > 0 else []
            id_to_logit = {tp.token_id: tp for tp in first_token_logits}
            answer_symbol_ids = log_entry.model_data.answer_symbol_ids
            symbol_to_logit = {
                symbol: id_to_logit[symbol_id]
                for symbol, symbol_id in answer_symbol_ids.items()
                if symbol_id in id_to_logit
            }
            if len(symbol_to_logit) > 0:
                extracted_pred = max(
                    symbol_to_logit, key=lambda x: symbol_to_logit[x].logprob
                )
                answer_format = AnswerFormat.PROPER
            label_logit = symbol_to_logit.get(label)
            log_probs = {
                "label": label_logit.logprob if label_logit else float("-inf"),
                "max_other_symbol": max(
                    (tp.logprob for k, tp in symbol_to_logit.items() if k != label),
                    default=float("-inf"),
                ),
                "max_other_token": max(
                    (
                        tp.logprob
                        for k, tp in id_to_logit.items()
                        if k != answer_symbol_ids[label]
                    ),
                    default=float("-inf"),
                ),
            }
        log_entry.stats = StatsRecord(
            label=label,
            prediction=extracted_pred,
            answer_format=answer_format,
            subject=log_entry.data.subject,
            log_probs=log_probs,
            scores=self._custom_scores(
                label,
                extracted_pred,
                ancillary_data=log_entry.data.ancillary_data,
            ),
        )
        return log_entry


class NextTokenLogGrader(LogGrader):
    """A log grader for multiple choice benchmarks that log the next token."""

    def __init__(
        self,
        output_processing_config: OutputProcessingConfig,
        recompute_stats: bool,
        answer_set: frozenset[str],
    ):
        """Initialize the next token log grader."""
        super().__init__(output_processing_config, recompute_stats)
        assert (
            len(answer_set) > 0
        ), "A non-empty answer set must be provided for next token log grader."
        self._answer_set = answer_set

    def _markup_entry_impl(self, log_entry: SampleRecord) -> SampleRecord:
        """Markup a single log entry with the computed statistics / scores."""
        label: Labeling | None = log_entry.data.label
        extracted_pred: Labeling | None = None
        answer_format = AnswerFormat.INVALID
        if (nt := log_entry.model_data.next_token) and (next_token := nt.output_text):
            match = re.findall(
                rf"\b({'|'.join(sorted(list(self._answer_set)))})\b", next_token
            )
            if len(match) > 0:
                extracted_pred, answer_format = match[0], AnswerFormat.PROPER
        log_entry.stats = StatsRecord(
            label=label,
            prediction=extracted_pred,
            answer_format=answer_format,
            subject=log_entry.data.subject,
            scores=self._custom_scores(
                label,
                extracted_pred,
                ancillary_data=log_entry.data.ancillary_data,
            ),
        )
        return log_entry


class ChatCompletionLogGrader(LogGrader):
    """A log grader for single-answer benchmarks that log full chat completions."""

    def __init__(
        self,
        output_processing_config: OutputProcessingConfig,
        model_format_config: PatternDef | None,
        recompute_stats: bool,
    ):
        """Initialize the chat completion log grader."""
        super().__init__(output_processing_config, recompute_stats)
        self._answer_matcher: Matcher[Any] = (
            SimpleMatcher(model_format_config)
            if model_format_config is not None
            else AllMatcher()
        )
        if answer_formats := output_processing_config.answer_formats:
            self._answer_matcher |= SequentialMatcher(*answer_formats)

    def _markup_entry_impl(self, log_entry: SampleRecord) -> SampleRecord:
        """Markup a single log entry with the computed statistics / scores."""
        label: Labeling | None = log_entry.data.label
        extracted_answer: Labeling | None = None
        answer_format = AnswerFormat.INVALID

        chat_comp = log_entry.model_data.chat_comp
        if chat_comp is not None and chat_comp.answer_text is not None:
            extracted_answer, answer_format = self._answer_matcher(
                chat_comp.answer_text
            )

        log_entry.stats = StatsRecord(
            label=label,
            prediction=extracted_answer,
            answer_format=answer_format,
            subject=log_entry.data.subject,
            num_output_tokens=(chat_comp.num_output_tokens or 0) if chat_comp else 0,
            max_token_halt=(chat_comp.max_token_halt or False) if chat_comp else False,
            scores=self._custom_scores(
                label,
                extracted_answer,
                ancillary_data=log_entry.data.ancillary_data,
            ),
        )
        return log_entry
