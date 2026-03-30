# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides `LogGrader` for grading / scoring benchmark logs."""

from abc import abstractmethod
from typing import Any

from faith._internal.iter.transform import IsoMapping
from faith._types.config.scoring import OutputProcessingConfig
from faith._types.record.sample import SampleRecord
from faith._types.record.stats import Labeling
from faith.benchmark.scores.domain_specific import DomainSpecificScore
from faith.benchmark.scores.scoring import Score


class LogGrader(IsoMapping[SampleRecord]):
    """Base class for log graders that process and grade benchmark logs."""

    def __init__(
        self,
        output_processing_config: OutputProcessingConfig,
    ):
        """Initialize the logs grader."""
        super().__init__()
        self._score_fns = DomainSpecificScore.from_configs(
            **output_processing_config.score_fns
        )

    def _map_fn(self, element: SampleRecord) -> SampleRecord:
        """Process a log record and annotate with its computed statistics / scores."""
        return self._markup_entry(element)

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
    def _markup_entry(self, log_entry: SampleRecord) -> SampleRecord:
        """Markup a single log entry with the computed statistics / scores."""
