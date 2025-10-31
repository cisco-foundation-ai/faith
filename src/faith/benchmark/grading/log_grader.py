# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides `LogGrader` for grading / scoring benchmark logs."""

import logging
from abc import abstractmethod
from typing import Any, Iterable

from tqdm import tqdm

from faith._internal.algo.matching import AnswerFormat
from faith._internal.iter.transform import IsoTransform
from faith._internal.metrics.domain_specific_scores import ScoreFn
from faith._internal.metrics.types import Labeling

logger = logging.getLogger(__name__)


class LogGrader(IsoTransform[dict[str, Any]]):
    """Base class for log graders that process and grade benchmark logs."""

    def __init__(
        self,
        output_processing_config: dict[str, Any],
        _model_format_config: dict[str, Any],
        recompute_stats: bool,
    ):
        """Initialize the logs grader."""
        super().__init__()
        self._recompute_stats = recompute_stats
        self._score_fns = ScoreFn.from_configs(
            **(output_processing_config.get("score_fns", None) or {})
        )

    def __call__(self, logs: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        """Process the logs and return the graded logs."""
        log_is_empty = True
        for log_entry in tqdm(
            logs, desc="Marking up logs", unit="entries", leave=False
        ):
            log_is_empty = False
            yield self._markup_entry(log_entry)
        if log_is_empty:
            logger.error("Benchmark logs are empty!")

    def _markup_entry(self, log_entry: dict[str, Any]) -> dict[str, Any]:
        """Markup a single log entry with the computed statistics / scores."""
        if "stats" not in log_entry or self._recompute_stats:
            return self._markup_entry_impl(log_entry)
        return self._normalize_entry(log_entry)

    def _custom_scores(
        self, label: Labeling, pred: Labeling | None
    ) -> dict[str, dict[str, float]]:
        """Return the custom scores defined in the output processing config."""
        if len(self._score_fns) == 0:
            return {}
        return {
            "scores": {
                name: score_fn(label, pred)
                for name, score_fn in self._score_fns.items()
            },
        }

    def _normalize_entry(self, log_entry: dict[str, Any]) -> dict[str, Any]:
        """Normalize the log entry to ensure consistent typing."""
        if "answer_format" in log_entry["stats"] and isinstance(
            log_entry["stats"]["answer_format"], str
        ):
            log_entry["stats"]["answer_format"] = AnswerFormat.from_string(
                log_entry["stats"]["answer_format"]
            )
        return log_entry

    @abstractmethod
    def _markup_entry_impl(self, log_entry: dict[str, Any]) -> dict[str, Any]:
        """Markup a single log entry with the computed statistics / scores."""
