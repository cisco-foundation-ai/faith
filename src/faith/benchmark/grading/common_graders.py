# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides common log graders for benchmarks."""

import re
from typing import Any

from faith._internal.algo.matching import (
    AllMatcher,
    AnswerFormat,
    Matcher,
    SequentialMatcher,
    SimpleMatcher,
)
from faith._internal.metrics.types import Labeling
from faith._internal.records.types import Record
from faith._internal.types.configs import Configuration
from faith.benchmark.grading.log_grader import LogGrader


class LogitsLogGrader(LogGrader):
    """A log grader for multiple choice benchmarks that log the next-token logits."""

    def _markup_entry_impl(self, log_entry: Record) -> Record:
        """Markup a single log entry with the computed statistics / scores."""
        label: str | None = log_entry["data"]["label"]
        extracted_pred: str | None = None
        answer_format = AnswerFormat.INVALID
        log_probs_stats = {}
        if label is not None and (logits := log_entry["model_data"].get("logits")):
            # TODO(https://github.com/cisco-foundation-ai/faith/issues/26):
            # Handle multiple logits entries; currently assumes only one entry.
            first_token_logits = logits[0] if len(logits) > 0 else []
            id_to_logit = {log["token_id"]: log for log in first_token_logits}
            answer_symbol_ids = log_entry["model_data"]["answer_symbol_ids"]
            symbol_to_logit = {
                symbol: id_to_logit[symbol_id]
                for symbol, symbol_id in answer_symbol_ids.items()
                if symbol_id in id_to_logit
            }
            if len(symbol_to_logit) > 0:
                extracted_pred = max(
                    symbol_to_logit, key=lambda x: symbol_to_logit[x]["logprob"]
                )
                answer_format = AnswerFormat.PROPER
            log_probs_stats = {
                "log_probs": {
                    "label": symbol_to_logit.get(label, {}).get(
                        "logprob", float("-inf")
                    ),
                    "max_other_symbol": max(
                        (
                            logit["logprob"]
                            for k, logit in symbol_to_logit.items()
                            if k != label
                        ),
                        default=float("-inf"),
                    ),
                    "max_other_token": max(
                        (
                            logit["logprob"]
                            for k, logit in id_to_logit.items()
                            if k != answer_symbol_ids[label]
                        ),
                        default=float("-inf"),
                    ),
                }
            }
        log_entry["stats"] = (
            log_probs_stats
            | {
                "label": label,
                "prediction": extracted_pred,
                "answer_format": answer_format,
                "subject": log_entry["data"].get("subject"),
            }
            | self._custom_scores(
                label,
                extracted_pred,
                ancillary_data=log_entry["data"].get("ancillary_data"),
            )
        )
        return log_entry


class NextTokenLogGrader(LogGrader):
    """A log grader for multiple choice benchmarks that log the next token."""

    def __init__(
        self,
        output_processing_config: Configuration,
        recompute_stats: bool,
        answer_set: frozenset[str],
    ):
        """Initialize the next token log grader."""
        super().__init__(output_processing_config, recompute_stats)
        assert (
            len(answer_set) > 0
        ), "A non-empty answer set must be provided for next token log grader."
        self._answer_set = answer_set

    def _markup_entry_impl(self, log_entry: Record) -> Record:
        """Markup a single log entry with the computed statistics / scores."""
        label: Labeling | None = log_entry["data"]["label"]
        extracted_pred: Labeling | None = None
        answer_format = AnswerFormat.INVALID
        if (
            next_token := log_entry["model_data"]
            .get("next_token", {})
            .get("output_text")
        ):
            match = re.findall(
                rf"\b({'|'.join(sorted(list(self._answer_set)))})\b", next_token
            )
            if len(match) > 0:
                extracted_pred, answer_format = match[0], AnswerFormat.PROPER
        log_entry["stats"] = {
            "label": label,
            "prediction": extracted_pred,
            "answer_format": answer_format,
            "subject": log_entry["data"].get("subject"),
        } | self._custom_scores(
            label,
            extracted_pred,
            ancillary_data=log_entry["data"].get("ancillary_data"),
        )
        return log_entry


class ChatCompletionLogGrader(LogGrader):
    """A log grader for single-answer benchmarks that log full chat completions."""

    def __init__(
        self,
        output_processing_config: Configuration,
        model_format_config: Configuration | None,
        recompute_stats: bool,
    ):
        """Initialize the chat completion log grader."""
        super().__init__(output_processing_config, recompute_stats)
        self._answer_matcher: Matcher[Any] = (
            SimpleMatcher(model_format_config)
            if model_format_config is not None
            else AllMatcher()
        )
        if answer_formats := output_processing_config.get("answer_formats"):
            self._answer_matcher |= SequentialMatcher(*answer_formats)

    def _markup_entry_impl(self, log_entry: Record) -> Record:
        """Markup a single log entry with the computed statistics / scores."""
        label: Labeling | None = log_entry["data"]["label"]
        extracted_answer: Labeling | None = None
        answer_format = AnswerFormat.INVALID

        chat_comp = log_entry["model_data"].get("chat_comp") or {}
        if (answer_text := chat_comp.get("answer_text")) is not None:
            extracted_answer, answer_format = self._answer_matcher(answer_text)

        log_entry["stats"] = {
            "label": label,
            "prediction": extracted_answer,
            "answer_format": answer_format,
            "subject": log_entry["data"].get("subject"),
            "num_output_tokens": chat_comp.get("num_output_tokens") or 0,
            "max_token_halt": chat_comp.get("max_token_halt") or False,
        } | self._custom_scores(
            label,
            extracted_answer,
            ancillary_data=log_entry["data"].get("ancillary_data"),
        )
        return log_entry
