# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Defines types and base classes for scoring functions used in benchmarks."""

from abc import ABC, abstractmethod
from typing import Any, Generic, NotRequired, Sequence, TypedDict, TypeVar

from faith._internal.metrics.types import Labeling

_LABELING = TypeVar("_LABELING", bound=Labeling)


class Score(TypedDict):
    """A base class representing a score and any associated metadata."""

    # The numeric score value given by a scoring function for a predicted answer.
    value: NotRequired[float]
    raw_value: float


class ScoreFn(ABC, Generic[_LABELING]):
    """A function that computes a score for a given predicted answer from its label."""

    def __init__(
        self,
        attributes: dict[str, Any] | None = None,
        score_range: dict[str, float] | None = None,
    ) -> None:
        """Initialize the ScoreFn."""
        self._attributes = attributes or {}
        score_range = score_range or {}
        self._min_score = score_range.get("min", 0.0)
        self._max_score = score_range.get("max", 1.0)
        assert (
            self._min_score < self._max_score
        ), "Invalid score range for judge: min {self._min_score} >= max {self._max_score}."

    @property
    def _raw_score_range(self) -> tuple[float, float]:
        """Get the raw score range for this scoring function."""
        return (0.0, 1.0)

    @property
    def attributes(self) -> dict[str, Any]:
        """Get the attributes associated with this scoring function."""
        return self._attributes

    def __call__(
        self, label: _LABELING, pred: _LABELING | None, **kwargs: Any
    ) -> Score:
        """Compute the score for a predicted answer against a given label.

        This score should be a non-negative float, where a higher score indicates a
        better match.
        """
        score = self._score(label, pred, **kwargs)
        score["value"] = self._rescale_score(score["raw_value"])
        return score

    def _rescale_score(self, raw_score: float) -> float:
        """Rescale the raw score to be within the configured score range."""
        raw_min, raw_max = self._raw_score_range
        assert (
            raw_min <= raw_score <= raw_max
        ), f"Raw score {raw_score} out of range [{raw_min}, {raw_max}]."
        return (self._max_score - self._min_score) * (raw_score - raw_min) / (
            raw_max - raw_min
        ) + self._min_score

    @abstractmethod
    def _score(self, label: _LABELING, pred: _LABELING | None, **kwargs: Any) -> Score:
        """Compute the score for a predicted answer against a given label.

        This is the implementation-specific scoring logic used by __call__.
        """

    @abstractmethod
    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of scores into a set of aggregate statistics."""
