# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Domain-specific scoring functions for evaluating short-answer model predictions."""

import logging
import math
from enum import Enum
from typing import Any, NotRequired, Sequence, Type, TypedDict

import numpy as np
from cvss import CVSS3, CVSSError
from jinja2 import Template

from faith._internal.algo.graph import wcc_dict
from faith._internal.algo.matching import AnswerFormat, SequentialMatcher
from faith._internal.parsing.expr import evaluate_expr
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.benchmark.scores.types import Score, ScoreFn
from faith.model.base import GenerationError
from faith.model.model_engine import ModelEngine

logger = logging.getLogger(__name__)


class CVSSScore(ScoreFn[str]):
    """A score for evaluating the correctness of CVSS vectors using their base score."""

    @property
    def _raw_score_range(self) -> tuple[float, float]:
        """Get the raw score range for this scoring function."""
        return (0.0, 10.0)

    def get_cvss_score(self, cvss_vector: str) -> float:
        """Get the base CVSS score from a CVSS vector string."""
        c = CVSS3(cvss_vector)
        return c.scores()[0]

    def _score(self, label: str, pred: str | None, **kwargs: Any) -> Score:
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
            return Score(raw_value=0.0)

        try:
            pred_score = self.get_cvss_score(pred)
        except CVSSError:
            return Score(raw_value=0.0)
        label_score = self.get_cvss_score(label)
        assert 0 <= pred_score <= 10, "Predicted CVSS score must be between 0 and 10."
        assert 0 <= label_score <= 10, "Label CVSS score must be between 0 and 10."
        return Score(raw_value=10.0 - abs(pred_score - label_score))

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of CVSS scores into statistics for the benchmark."""
        score_values = [s["value"] for s in scores]
        return {
            "mean": float(np.mean(score_values)),
            "median": float(np.median(score_values)),
        }


class JaccardIndex(ScoreFn[tuple[str, ...]]):
    """A score for evaluating the correctness between two sets of labels."""

    def _score(
        self,
        label: tuple[str, ...] | None,
        pred: tuple[str, ...] | None,
        **kwargs: Any,
    ) -> Score:
        """Compute the Jaccard score between two sets of labels."""
        label_set = set(label or [])
        pred_set = set(pred or [])

        return (
            Score(raw_value=len(label_set & pred_set) / len(label_set | pred_set))
            if label_set or pred_set
            else Score(raw_value=1.0)
        )

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of Jaccard scores into statistics for the benchmark."""
        score_values = [s["value"] for s in scores]
        return {
            "mean": float(np.mean(score_values)),
            "median": float(np.median(score_values)),
        }


class LogScaledScore(ScoreFn[str]):
    """A score for evaluating numeric answers with a logarithmically scaled score."""

    def __init__(
        self,
        tolerance: float,
        scaling: float = 10.0,
        attributes: dict[str, Any] | None = None,
        score_range: dict[str, float] | None = None,
    ) -> None:
        """Initialize the LogScaledScore with a given tolerance."""
        super().__init__(attributes=attributes, score_range=score_range)
        assert 0 < tolerance < 1, "Tolerance must be in (0, 1)."
        assert scaling > 0, "Scaling must be positive."
        self._tolerance = tolerance
        self._scaling = scaling

    def _score(self, label: str, pred: str | None, **kwargs: Any) -> Score:
        """Compute the numeric answer score between a label and a prediction."""
        if pred is None:
            return Score(raw_value=0.0)

        try:
            label_value = float(label)
            pred_value = float(pred)
        except ValueError:
            return Score(raw_value=0.0)

        tol = self._tolerance * (abs(label_value) if label_value != 0 else 1)
        tol_error = abs(label_value - pred_value) / tol
        return Score(
            raw_value=max(0, 1 - math.log(1 + tol_error) / math.log(1 + self._scaling))
        )

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of alias accuracy scores into statistics for the benchmark."""
        return {"mean": float(np.mean([s["value"] for s in scores]))}


class AliasAccuracyScore(ScoreFn[str]):
    """A score for evaluating accuracy in predicting a group of aliases."""

    def __init__(
        self,
        alias_map: dict[str, list[str]],
        attributes: dict[str, Any] | None = None,
        score_range: dict[str, float] | None = None,
    ) -> None:
        """Initialize the AliasAccuracyScore with a dictionary of aliases."""
        super().__init__(attributes=attributes, score_range=score_range)
        self._alias_wcc = wcc_dict(alias_map)

    def _score(self, label: str, pred: str | None, **kwargs: Any) -> Score:
        """Evaluate the connection between two threat actors."""
        if pred is None:
            return Score(raw_value=0.0)

        normalized_label = label.strip().lower()
        normalized_pred = pred.strip().lower()

        label_alias_wcc = self._alias_wcc.get(normalized_label, -1)
        assert label_alias_wcc != -1, f"Label '{label}' not found in alias dictionary."
        pred_alias_wcc = self._alias_wcc.get(normalized_pred, -1)

        return Score(raw_value=1.0 if label_alias_wcc == pred_alias_wcc else 0.0)

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

    summary_details: dict[str, Any]
    full_response: str


class LLMJudgeScore(ScoreFn[str]):
    """An LLM-based judge for scoring long-answer responses."""

    def __init__(
        self,
        judge_prompt_template: str,
        judge_model: dict[str, Any],
        verdict_formats: list[dict[str, Any]],
        llm_score_range: dict[str, float] | None = None,
        attributes: dict[str, Any] | None = None,
        score_range: dict[str, float] | None = None,
    ) -> None:
        """Initialize the LLM-based judge."""
        super().__init__(attributes=attributes, score_range=score_range)
        self._judge_prompt_template = Template(judge_prompt_template)
        llm_score_range = llm_score_range or {}
        self._llm_min_score = llm_score_range.get("min", 0.0)
        self._llm_max_score = llm_score_range.get("max", 1.0)
        assert (
            self._llm_min_score < self._llm_max_score
        ), "Invalid score range for judge: min {self._llm_min_score} >= max {self._llm_max_score}."
        model_engine = ModelEngine.from_string(judge_model["model_engine"])
        self._judge_model = model_engine.create_model(
            judge_model["model_path"], **judge_model.get("engine_kwargs", {})
        )
        self._judge_model_formatter = PromptFormatter.CHAT
        self._judge_generation_kwargs = judge_model.get("generation_kwargs", {})
        self._verdict_matcher = SequentialMatcher(*verdict_formats)

    @property
    def _raw_score_range(self) -> tuple[float, float]:
        """Get the raw score range for this scoring function."""
        return (self._llm_min_score, self._llm_max_score)

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

    def _score(
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
        match_format = AnswerFormat.INVALID
        num_tries = 0
        while match_format == AnswerFormat.INVALID and num_tries < 5:
            verdict = self._query_judge_model(judge_prompt)
            try:
                verdict_dict, match_format = self._verdict_matcher(verdict)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning("Error parsing judge verdict, retrying.\n\n%s", verdict)
            num_tries += 1
        if match_format == AnswerFormat.INVALID:
            try:
                verdict_dict, match_format = self._verdict_matcher(verdict)
            except Exception as e:
                raise RuntimeError(f"Error parsing judge verdict:\n\n{verdict}") from e
        assert (
            match_format != AnswerFormat.INVALID
        ), f"Could not parse judge verdict:\n\n{verdict}"
        parsed_verdict: _ParsedVerdict = verdict_dict
        return LLMJudgeVerdict(
            raw_value=parsed_verdict["awarded_points"],
            summary_details=parsed_verdict.get("details", {}),
            full_response=verdict,
        )

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of judgement grades into the statistics for the benchmark."""
        per_question_grades = [s["value"] for s in scores]
        return {
            "mean": np.mean(per_question_grades),
            "median": np.median(per_question_grades),
            "stddev": np.std(per_question_grades),
        }


class SubScores(Score):
    """A TypedDict representing sub-scores from multiple scoring functions."""

    sub_scores: dict[str, Score]


class CompositeScore(ScoreFn[Any]):
    """A composite score function that combines multiple scoring functions."""

    _RESERVED_NAMES = {"self"}

    def __init__(
        self,
        aggregation: str,
        attributes: dict[str, Any] | None = None,
        score_range: dict[str, float] | None = None,
        **score_fn_configs: dict[str, Any],
    ) -> None:
        """Initialize the CompositeScore with a dictionary of scoring functions."""
        super().__init__(attributes=attributes, score_range=score_range)
        self._aggregation = aggregation
        self._score_fns = DomainSpecificScore.from_configs(**score_fn_configs)
        assert set(self._score_fns.keys()).isdisjoint(
            self._RESERVED_NAMES
        ), f"Score function names cannot be any of the reserved names: {self._RESERVED_NAMES}"

    def _score(self, label: Any, pred: Any, **kwargs: Any) -> Score:
        """Compute the composite score for a label and prediction from their sub-scores."""
        attributes = {
            score_name: score_fn.attributes
            for score_name, score_fn in self._score_fns.items()
        }
        sub_scores = {
            score_name: score_fn(label, pred, **kwargs)
            for score_name, score_fn in self._score_fns.items()
        }
        scores = {name: sub_score["value"] for name, sub_score in sub_scores.items()}
        return SubScores(
            raw_value=evaluate_expr(
                self._aggregation, names={"attrs": attributes, "scores": scores}
            ),
            sub_scores=sub_scores,
        )

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Aggregate a list of composite scores into statistics for the benchmark."""
        return {"mean": float(np.mean([s["value"] for s in scores]))}


class DomainSpecificScore(Enum):
    """Enum for scores used in domain-specific benchmarks."""

    CVSS = (CVSSScore,)  # Score from CVSS vectors, normalized to [0, 1].
    JACCARD = (
        JaccardIndex,
    )  # Score from Jaccard index between sets of labels; in [0, 1].
    LOG_SCALED_SCORE = (LogScaledScore,)  # Score for numeric answers.
    ALIAS_ACCURACY = (AliasAccuracyScore,)  # Accuracy score for alias matching.
    LLM_JUDGE = (LLMJudgeScore,)  # Score from LLM-based judge evaluation.
    COMPOSITE = (CompositeScore,)  # Composite score from multiple sub-scores.

    def __init__(self, scoring_cls: Type[ScoreFn]) -> None:
        """Initialize the DomainSpecificScore with the enum value's scoring class."""
        self._scoring_cls = scoring_cls

    def __str__(self) -> str:
        """Return the name of the score function."""
        return self.name.lower()

    @staticmethod
    def from_string(name: str) -> "DomainSpecificScore":
        """Get the DomainSpecificScore instance from its string representation."""
        try:
            return DomainSpecificScore[name.upper()]
        except KeyError as e:
            raise ValueError(
                f"Invalid score function name: {name}. Available options: {[m.name for m in DomainSpecificScore]}"
            ) from e

    def get_score_fn(self, **kwargs: dict[str, Any]) -> ScoreFn:
        """Get the scorer instance for this score function."""
        return self._scoring_cls(**kwargs)

    @staticmethod
    def from_configs(**score_fn_kwargs: dict[str, Any]) -> dict[str, ScoreFn]:
        """Load custom score functions using the config supplied by each key-word argument."""
        return {
            name: DomainSpecificScore.from_string(score_cfg["type"]).get_score_fn(
                **{k: v for k, v in score_cfg.items() if k != "type"}
            )
            for name, score_cfg in score_fn_kwargs.items()
        }
