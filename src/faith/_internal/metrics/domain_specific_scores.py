# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Domain-specific scoring functions for evaluating short-answer model predictions."""

import math
from enum import Enum
from typing import Any, NotRequired, Protocol, Sequence, Type, TypedDict

import numpy as np
from cvss import CVSS3, CVSSError
from jinja2 import Template

from faith._internal.algo.graph import wcc_dict
from faith._internal.algo.matching import AnswerFormat, SequentialMatcher
from faith._internal.metrics.types import Labeling
from faith._internal.parsing.expr import evaluate_expr
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import GenerationError
from faith.model.model_engine import ModelEngine


class Score(TypedDict):
    """A base class representing a score and any associated metadata."""

    # The numeric score value given by a scoring function for a predicted answer.
    value: float


class AnswerScoreFn(Protocol):
    """A function that computes a score for a given predicted answer from its label."""

    def __call__(self, label: Labeling, pred: Labeling | None, **kwargs: Any) -> Score:
        """Compute the score for a predicted answer against a given label.

        This score should be a non-negative float, where a higher score indicates a
        better match.
        """

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of scores into a set of aggregate statistics."""


class CVSSScore:
    """A score for evaluating the correctness of CVSS vectors using their base score."""

    def get_cvss_score(self, cvss_vector: str) -> float:
        """Get the base CVSS score from a CVSS vector string."""
        c = CVSS3(cvss_vector)
        return c.scores()[0] / 10.0

    def __call__(self, label: str, pred: str | None, **kwargs: Any) -> Score:
        """Compute the CVSS score for a predicted CVSS vector against a label.

        This score computes the absolute deviation between the predicted CVSS score
        and the ground truth CVSS score. It returns a value between 0 and 1, where 1
        indicates a perfect match. A score of 0 is returned for invalid predictions.

        Args:
            label (str): The ground truth CVSS vector.
            pred (str): The predicted CVSS vector.
            kwargs: Additional keyword arguments (not used).

        Returns:
            Score: The CVSS score, normalized to [0, 1]. A score of 1.0 indicates a
            perfect match.
        """
        if pred is None:
            return Score(value=0.0)

        try:
            pred_score = self.get_cvss_score(pred)
        except CVSSError:
            return Score(value=0.0)
        label_score = self.get_cvss_score(label)
        assert 0 <= pred_score <= 1, "Predicted CVSS score must be between 0 and 1."
        assert 0 <= label_score <= 1, "Label CVSS score must be between 0 and 1."
        return Score(value=1.0 - abs(pred_score - label_score))

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of CVSS scores into statistics for the benchmark."""
        score_values = [s["value"] for s in scores]
        return {
            "mean": float(np.mean(score_values)),
            "median": float(np.median(score_values)),
        }


class JaccardIndex:
    """A score for evaluating the correctness between two sets of labels."""

    def __call__(
        self,
        label: tuple[str, ...] | None,
        pred: tuple[str, ...] | None,
        **kwargs: Any,
    ) -> Score:
        """Compute the Jaccard score between two sets of labels."""
        label_set = set(label or [])
        pred_set = set(pred or [])

        return (
            Score(value=len(label_set & pred_set) / len(label_set | pred_set))
            if label_set or pred_set
            else Score(value=1.0)
        )

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of Jaccard scores into statistics for the benchmark."""
        score_values = [s["value"] for s in scores]
        return {
            "mean": float(np.mean(score_values)),
            "median": float(np.median(score_values)),
        }


class LogScaledScore:
    """A score for evaluating numeric answers with a logarithmically scaled score."""

    def __init__(self, tolerance: float, scaling: float = 10.0) -> None:
        """Initialize the LogScaledScore with a given tolerance."""
        assert 0 < tolerance < 1, "Tolerance must be in (0, 1)."
        assert scaling > 0, "Scaling must be positive."
        self._tolerance = tolerance
        self._scaling = scaling

    def __call__(self, label: str, pred: str | None, **kwargs: Any) -> Score:
        """Compute the numeric answer score between a label and a prediction."""
        if pred is None:
            return Score(value=0.0)

        try:
            label_value = float(label)
            pred_value = float(pred)
        except ValueError:
            return Score(value=0.0)

        tol = self._tolerance * (abs(label_value) if label_value != 0 else 1)
        tol_error = abs(label_value - pred_value) / tol
        return Score(
            value=max(0, 1 - math.log(1 + tol_error) / math.log(1 + self._scaling))
        )

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of alias accuracy scores into statistics for the benchmark."""
        return {"mean": float(np.mean([s["value"] for s in scores]))}


class AliasAccuracyScore:
    """A score for evaluating accuracy in predicting a group of aliases."""

    def __init__(self, alias_map: dict[str, list[str]]) -> None:
        """Initialize the AliasAccuracyScore with a dictionary of aliases."""
        self._alias_wcc = wcc_dict(alias_map)

    def __call__(self, label: str, pred: str | None, **kwargs: Any) -> Score:
        """Evaluate the connection between two threat actors."""
        if pred is None:
            return Score(value=0.0)

        normalized_label = label.strip().lower()
        normalized_pred = pred.strip().lower()

        label_alias_wcc = self._alias_wcc.get(normalized_label, -1)
        assert label_alias_wcc != -1, f"Label '{label}' not found in alias dictionary."
        pred_alias_wcc = self._alias_wcc.get(normalized_pred, -1)

        return Score(value=1.0 if label_alias_wcc == pred_alias_wcc else 0.0)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of alias accuracy scores into statistics for the benchmark."""
        return {"accuracy": float(np.mean([s["value"] for s in scores]))}


class _ParsedVerdict(TypedDict):
    """A TypedDict representing the parsed verdict from an LLM-based judge.

    This is an internal structure used for type checking from the parsed LLM response.
    """

    awarded_points: float
    details: NotRequired[dict[str, Any]]


class LLMJudgeVerdict(Score):
    """A TypedDict representing the full verdict from an LLM-based judge."""

    raw_value: float
    min_value: float
    max_value: float
    summary_details: dict[str, Any]
    full_response: str


class LLMJudgeScore:
    """An LLM-based judge for scoring long-answer responses."""

    def __init__(
        self,
        judge_prompt_template: str,
        judge_model: dict[str, Any],
        verdict_formats: list[dict[str, Any]],
        score_range: dict[str, float] | None = None,
    ) -> None:
        """Initialize the LLM-based judge."""
        self._judge_prompt_template = Template(judge_prompt_template)
        score_range = score_range or {}
        self._min_score = score_range.get("min", 0.0)
        self._max_score = score_range.get("max", 1.0)
        assert (
            self._min_score < self._max_score
        ), "Invalid score range for judge: min {self._min_score} >= max {self._max_score}."
        model_engine = ModelEngine.from_string(judge_model["model_engine"])
        self._judge_model = model_engine.create_model(
            judge_model["model_path"], **judge_model.get("engine_kwargs", {})
        )
        self._judge_model_formatter = PromptFormatter.CHAT
        self._judge_generation_kwargs = judge_model.get("generation_kwargs", {})
        self._verdict_matcher = SequentialMatcher(*verdict_formats)

    def _query_judge_model(self, prompt: str) -> str:
        """Prompt the judge model with an evaluation prompt and return the response."""
        response = next(
            iter(
                self._judge_model.query(
                    [
                        self._judge_model_formatter.format(
                            system_prompt=None, prompt=prompt, response_leadin=None
                        )
                    ],
                    **self._judge_generation_kwargs,
                )
            )
        )
        if isinstance(response, GenerationError):
            raise RuntimeError(
                f"Judge model generation error: {response.error_message}"
            )
        return response.answer_text or ""

    def __call__(
        self,
        label: str,
        pred: str | None,
        ancillary_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMJudgeVerdict:
        """Compute the score for the predicted answer based on the judge's evaluation."""
        judge_prompt = self._judge_prompt_template.render(
            correct_answer=label,
            generated_answer=pred,
            ancillary_data=ancillary_data or {},
        )
        verdict = self._query_judge_model(judge_prompt)
        verdict_dict, match_format = self._verdict_matcher(verdict)
        assert (
            match_format != AnswerFormat.INVALID
        ), f"Could not parse judge verdict:\n\n{verdict}"
        parsed_verdict: _ParsedVerdict = verdict_dict
        return LLMJudgeVerdict(
            value=(parsed_verdict["awarded_points"] - self._min_score)
            / (self._max_score - self._min_score),
            raw_value=parsed_verdict["awarded_points"],
            min_value=self._min_score,
            max_value=self._max_score,
            summary_details=parsed_verdict.get("details", {}),
            full_response=verdict,
        )

    def aggregate(self, judgements: Sequence[LLMJudgeVerdict]) -> dict[str, float]:
        """Aggregate a list of judgement grades into the statistics for the benchmark."""
        per_question_grades = [s["value"] for s in judgements]
        return {
            "mean": np.mean(per_question_grades),
            "median": np.median(per_question_grades),
            "stddev": np.std(per_question_grades),
        }


class SubScores(Score):
    """A TypedDict representing sub-scores from multiple scoring functions."""

    sub_scores: dict[str, Score]


class CompositeScore:
    """A composite score function that combines multiple scoring functions."""

    def __init__(self, reduce_expr: str, **score_fn_configs: dict[str, Any]) -> None:
        """Initialize the CompositeScore with a dictionary of scoring functions."""
        self._reduce_expr = reduce_expr
        self._score_fns = ScoreFn.from_configs(**score_fn_configs)

    def __call__(self, label: Any, pred: Any, **kwargs: Any) -> Score:
        """Compute the composite score for a label and prediction from their sub-scores."""
        sub_scores = {
            score_name: score_fn(label, pred, **kwargs)
            for score_name, score_fn in self._score_fns.items()
        }
        scores = {name: sub_score["value"] for name, sub_score in sub_scores.items()}
        return SubScores(
            value=evaluate_expr(self._reduce_expr, names={"scores": scores}),
            sub_scores=sub_scores,
        )

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of composite scores into statistics for the benchmark."""
        return {"mean": float(np.mean([s["value"] for s in scores]))}


class ScoreFn(Enum):
    """Enum for score functions used in domain-specific benchmarks."""

    CVSS = (CVSSScore,)  # Score from CVSS vectors, normalized to [0, 1].
    JACCARD = (
        JaccardIndex,
    )  # Score from Jaccard index between sets of labels; in [0, 1].
    LOG_SCALED_SCORE = (LogScaledScore,)  # Score for numeric answers.
    ALIAS_ACCURACY = (AliasAccuracyScore,)  # Accuracy score for alias matching.
    LLM_JUDGE = (LLMJudgeScore,)  # Score from LLM-based judge evaluation.
    COMPOSITE = (CompositeScore,)  # Composite score from multiple sub-scores.

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
        except KeyError as e:
            raise ValueError(
                f"Invalid score function name: {name}. Available options: {[m.name for m in ScoreFn]}"
            ) from e

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
