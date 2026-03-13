# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Defines `GradeAggregator` for aggregating benchmark grades from benchmark logs."""

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import fields
from typing import Any

import numpy as np

from faith._internal.iter.transform import Reducer
from faith._internal.types.configs import Configuration
from faith._internal.types.stats import MetricSummary
from faith._internal.types.validation import assert_same_length
from faith._types.records.stats import StatsRecord
from faith.benchmark.scores.domain_specific import DomainSpecificScore
from faith.benchmark.scores.types import Score


class GradeAggregator(Reducer[StatsRecord | None, MetricSummary]):
    """Base class for aggregating benchmark grades from benchmark logs."""

    def __init__(self, output_processing_config: Configuration) -> None:
        """Initialize the GradeAggregator."""
        super().__init__()
        self._score_fns = DomainSpecificScore.from_configs(
            **(output_processing_config.get("score_fns") or {})
        )

    def __call__(self, stats_logs: Iterable[StatsRecord | None]) -> MetricSummary:
        """Reduce the collected statistics to their overall benchmark metrics."""
        test_stats = GradeAggregator._stats_transpose(stats_logs)
        logit_stats = (
            GradeAggregator._logits_stats(test_stats["log_probs"])
            if "log_probs" in test_stats
            else {}
        )
        return logit_stats | self._aggregate(**test_stats)

    @staticmethod
    def _stats_transpose(
        stats_logs: Iterable[StatsRecord | None],
    ) -> dict[str, Sequence[Any]]:
        """Transpose the 'stats' dataclass fields in `logs` to a dictionary of lists."""
        transposed_stats: dict[str, list[Any]] = defaultdict(list)
        for stats in stats_logs:
            for key in ({f.name for f in fields(stats)} if stats else set()):
                transposed_stats[key].append(getattr(stats, key))
        return {
            k: v for k, v in transposed_stats.items() if any(x is not None for x in v)
        }

    @staticmethod
    def _logits_stats(
        log_prob_stats: Sequence[dict[str, float | None]],
    ) -> MetricSummary:
        # Aggregate log probabilities for each label.
        stats_vectors = defaultdict(list)
        for log_prob in log_prob_stats:
            for stat, value in log_prob.items():
                stats_vectors[stat].append(value)
        label_lp = np.array(stats_vectors.get("label") or [], dtype=float)
        mos_lp = np.array(stats_vectors.get("max_other_symbol") or [], dtype=float)
        mot_lp = np.array(stats_vectors.get("max_other_token") or [], dtype=float)

        os_is_finite = np.isfinite(mos_lp) & np.isfinite(label_lp)
        exceeds_os = (label_lp > mos_lp) & os_is_finite
        beneath_os = (mos_lp > label_lp) & os_is_finite
        ot_is_finite = np.isfinite(mot_lp) & np.isfinite(label_lp)
        exceeds_ot = (label_lp > mot_lp) & ot_is_finite
        beneath_ot = (mot_lp > label_lp) & ot_is_finite
        return {
            "perplexity": np.exp(-np.mean(label_lp[np.isfinite(label_lp)])),
            "mean_confidence_gap": {
                "correct_symbol": np.mean(label_lp[exceeds_os] - mos_lp[exceeds_os]),
                "correct_token": np.mean(label_lp[exceeds_ot] - mot_lp[exceeds_ot]),
                "incorrect_symbol": np.mean(mos_lp[beneath_os] - label_lp[beneath_os]),
                "incorrect_token": np.mean(mot_lp[beneath_ot] - label_lp[beneath_ot]),
            },
            "num_near_ties": {
                "symbol": np.sum(
                    np.isclose(label_lp[os_is_finite], mos_lp[os_is_finite])
                ),
                "token": np.sum(
                    np.isclose(label_lp[ot_is_finite], mot_lp[ot_is_finite])
                ),
            },
        }

    def _aggregate_scores(self, scores: Sequence[dict[str, Score]]) -> dict[str, float]:
        # Convert the scores into a dictionary of lists for each score function.
        scores_dict: dict[str, list[Score]] = {n: [] for n in self._score_fns.keys()}
        for score in scores:
            for name, value in score.items():
                assert (
                    name in self._score_fns
                ), f"Score '{name}' is not defined in the benchmark's score functions: {list(self._score_fns.keys())}."
                scores_dict[name].append(value)
        assert_same_length(expected_length=len(scores), **scores_dict)

        # Aggregate the scores for each score function using its aggregation methods.
        return {
            f"{agg_name}_{name}": agg_value
            for name, values in scores_dict.items()
            for agg_name, agg_value in self._score_fns[name].aggregate(values).items()
        }

    @abstractmethod
    def _aggregate(self, **kwargs: Sequence[Any]) -> MetricSummary:
        """Aggregate the overall benchmark metrics from the collected statistics."""
