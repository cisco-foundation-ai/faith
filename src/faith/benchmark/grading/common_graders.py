# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides common log graders for benchmarks."""
import re
from typing import Any

from faith._internal.algo.matching import AnswerFormat
from faith._internal.metrics.types import Labeling
from faith.benchmark.grading.log_grader import LogGrader


class LogitsLogGrader(LogGrader):
    """A log grader for multiple choice benchmarks that log the next-token logits."""

    def _markup_entry_impl(self, log_entry: dict[str, Any]) -> dict[str, Any]:
        """Markup a single log entry with the computed statistics / scores."""
        label: Labeling = log_entry["data"]["label"]
        extracted_pred: Labeling | None = None
        answer_format = AnswerFormat.INVALID
        log_probs_stats = {}
        if logits := log_entry["model_data"].get("logits", None):
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
                "subject": log_entry["data"].get("subject", None),
            }
            | self._custom_scores(label, extracted_pred)
        )
        return log_entry


class NextTokenLogGrader(LogGrader):
    """A log grader for multiple choice benchmarks that log the next token."""

    def __init__(
        self,
        output_processing_config: dict[str, Any],
        model_format_config: dict[str, Any],
        recompute_stats: bool,
        answer_set: frozenset[str],
    ):
        """Initialize the next token log grader."""
        super().__init__(output_processing_config, model_format_config, recompute_stats)
        assert (
            len(answer_set) > 0
        ), "A non-empty answer set must be provided for next token log grader."
        self._answer_set = answer_set

    def _markup_entry_impl(self, log_entry: dict[str, Any]) -> dict[str, Any]:
        """Markup a single log entry with the computed statistics / scores."""
        label: Labeling = log_entry["data"]["label"]
        extracted_pred: Labeling | None = None
        answer_format = AnswerFormat.INVALID
        if (
            next_token := log_entry["model_data"]
            .get("next_token", {})
            .get("output_text", None)
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
            "subject": log_entry["data"].get("subject", None),
        } | self._custom_scores(label, extracted_pred)
        return log_entry


class ChatCompletionLogGrader(LogGrader):
    """A log grader for single-answer benchmarks that log full chat completions."""

    def _markup_entry_impl(self, log_entry: dict[str, Any]) -> dict[str, Any]:
        """Markup a single log entry with the computed statistics / scores."""
        label: Labeling = log_entry["data"]["label"]
        extracted_pred: Labeling | None = None
        answer_format = AnswerFormat.INVALID
        if (
            output_text := log_entry["model_data"]
            .get("chat_comp", {})
            .get("output_text", None)
        ):
            extracted_pred, answer_format = self._answer_matcher(output_text)

        log_entry["stats"] = {
            "label": label,
            "prediction": extracted_pred,
            "answer_format": answer_format,
            "subject": log_entry["data"].get("subject", None),
            "num_output_tokens": log_entry["model_data"]
            .get("chat_comp", {})
            .get("num_output_tokens", 0),
            "max_token_halt": log_entry["model_data"]
            .get("chat_comp", {})
            .get("max_token_halt", False),
        } | self._custom_scores(label, extracted_pred)
        return log_entry
