# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Domain-specific scoring functions for evaluating short-answer model predictions."""
from enum import Enum
from typing import Any, Protocol, Sequence, Type

import numpy as np
from cvss import CVSS3

from faith._internal.algo.graph import wcc_dict
from faith._internal.metrics.types import Labeling


class AnswerScoreFn(Protocol):
    """A function that computes a score for a given predicted answer from its label."""

    def __call__(self, label: Labeling, pred: Labeling | None) -> float:
        """Compute the score for a predicted answer against a given label.

        This score should be a non-negative float, where a higher score indicates a better match.
        """

    def aggregate(self, scores: Sequence[float]) -> dict[str, float]:
        """Aggregate a list of scores into a set of aggregate statistics."""


class CVSSScore:
    """A score for evaluating the correctness of CVSS vectors using their base score."""

    def get_cvss_score(self, cvss_vector: str) -> float:
        """Get the base CVSS score from a CVSS vector string."""
        c = CVSS3(cvss_vector)
        return c.scores()[0] / 10.0

    def __call__(self, label: str, pred: str | None) -> float:
        """Compute the CVSS score for a predicted CVSS vector against a label.

        This score computes the absolute deviation between the predicted CVSS score
        and the ground truth CVSS score. It returns a value between 0 and 1, where 1
        indicates a perfect match. A score of 0 is returned for invalid predictions.

        Args:
            pred (str): The predicted CVSS vector.
            label (str): The ground truth CVSS vector.

        Returns:
            float: The CVSS score, normalized to [0, 1]. A score of 1.0 indicates a perfect match.
        """
        if pred is None:
            return 0.0

        try:
            pred_score = self.get_cvss_score(pred)
        except Exception:
            return 0.0
        label_score = self.get_cvss_score(label)
        assert 0 <= pred_score <= 1, "Predicted CVSS score must be between 0 and 1."
        assert 0 <= label_score <= 1, "Label CVSS score must be between 0 and 1."
        return 1.0 - abs(pred_score - label_score)

    def aggregate(self, scores: Sequence[float]) -> dict[str, float]:
        """Aggregate a list of CVSS scores into statistics for the benchmark."""
        return {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
        }


class JaccardIndex:
    """A score for evaluating the correctness between two sets of labels."""

    def __call__(
        self, label: tuple[str, ...] | None, pred: tuple[str, ...] | None
    ) -> float:
        """Compute the Jaccard score between two sets of labels."""
        label_set = set(label or [])
        pred_set = set(pred or [])

        return (
            len(label_set & pred_set) / len(label_set | pred_set)
            if label_set or pred_set
            else 1.0
        )

    def aggregate(self, scores: Sequence[float]) -> dict[str, float]:
        """Aggregate a list of Jaccard scores into statistics for the benchmark."""
        return {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
        }


class AliasAccuracyScore:
    """A score for evaluating accuracy in predicting a group of aliases."""

    def __init__(self, alias_map: dict[str, list[str]]) -> None:
        """Initialize the AliasAccuracyScore with a dictionary of aliases."""
        self._alias_wcc = wcc_dict(alias_map)

    def __call__(self, label: str, pred: str | None) -> float:
        """Evaluate the connection between two threat actors."""
        if pred is None:
            return 0.0

        normalized_label = label.strip().lower()
        normalized_pred = pred.strip().lower()

        label_alias_wcc = self._alias_wcc.get(normalized_label, -1)
        assert label_alias_wcc != -1, f"Label '{label}' not found in alias dictionary."
        pred_alias_wcc = self._alias_wcc.get(normalized_pred, -1)

        return 1.0 if label_alias_wcc == pred_alias_wcc else 0.0

    def aggregate(self, scores: Sequence[float]) -> dict[str, float]:
        """Aggregate a list of alias accuracy scores into statistics for the benchmark."""
        return {"accuracy": float(np.mean(scores))}


class ScoreFn(Enum):
    """Enum for score functions used in domain-specific benchmarks."""

    CVSS = (CVSSScore,)  # Score from CVSS vectors, normalized to [0, 1].
    JACCARD = (
        JaccardIndex,
    )  # Score from Jaccard index between sets of labels; in [0, 1].
    ALIAS_ACCURACY = (AliasAccuracyScore,)  # Accuracy score for alias matching.

    def __init__(self, scoring_cls: Type[AnswerScoreFn]) -> None:
        """Initialize the ScoreFn with the enum value's scoring class."""
        self._scoring_cls = scoring_cls

    def __str__(self) -> str:
        """Return the name of the score function."""
        return self.name.lower()

    @staticmethod
    def from_string(name: str) -> "ScoreFn":
        """Get the ScoreFn instance from its string representation."""
        try:
            return ScoreFn[name.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid score function name: {name}. Available options: {[m.name for m in ScoreFn]}"
            )

    def get_score_fn(self, **kwargs: dict[str, Any]) -> AnswerScoreFn:
        """Get the scorer instance for this score function."""
        return self._scoring_cls(**kwargs)

    @staticmethod
    def from_configs(**score_fn_kwargs: dict[str, Any]) -> dict[str, AnswerScoreFn]:
        """Load custom score functions using the config supplied by each key-word argument."""
        return {
            name: ScoreFn.from_string(score_cfg["type"]).get_score_fn(
                **{k: v for k, v in score_cfg.items() if k != "type"}
            )
            for name, score_cfg in score_fn_kwargs.items()
        }
