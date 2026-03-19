# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides `LogGrader` for grading / scoring benchmark logs."""

import logging
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

from tqdm import tqdm

from faith._internal.iter.transform import IsoTransform
from faith._types.config.scoring import OutputProcessingConfig
from faith._types.record.sample_record import SampleRecord
from faith._types.record.stats import Labeling
from faith.benchmark.scores.domain_specific import DomainSpecificScore
from faith.benchmark.scores.scoring import Score

logger = logging.getLogger(__name__)


class LogGrader(IsoTransform[SampleRecord]):
    """Base class for log graders that process and grade benchmark logs."""

    def __init__(
        self,
        output_processing_config: OutputProcessingConfig,
        recompute_stats: bool,
    ):
        """Initialize the logs grader."""
        super().__init__()
        self._recompute_stats = recompute_stats
        self._score_fns = DomainSpecificScore.from_configs(
            **output_processing_config.score_fns
        )

    def __call__(self, logs: Iterable[SampleRecord]) -> Iterable[SampleRecord]:
        """Process the logs and return the graded logs."""
        log_is_empty = True
        for log_entry in tqdm(
            logs, desc="Marking up logs", unit="entries", leave=False
        ):
            log_is_empty = False
            yield self._markup_entry(log_entry)
        if log_is_empty:
            logger.error("Benchmark logs are empty!")

    def _markup_entry(self, log_entry: SampleRecord) -> SampleRecord:
        """Markup a single log entry with the computed statistics / scores."""
        if not log_entry.stats or self._recompute_stats:
            return self._markup_entry_impl(log_entry)
        return log_entry

    def _custom_scores(
        self, label: Labeling | None, pred: Labeling | None, **kwargs: Any
    ) -> dict[str, Score] | None:
        """Return the custom scores defined in the output processing config."""
        if len(self._score_fns) == 0:
            return None
        return {
            name: score_fn(label, pred, **kwargs)
            for name, score_fn in self._score_fns.items()
        }

    @abstractmethod
    def _markup_entry_impl(self, log_entry: SampleRecord) -> SampleRecord:
        """Markup a single log entry with the computed statistics / scores."""
