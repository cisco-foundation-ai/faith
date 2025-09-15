# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""This module provides functions to compute common metrics for scoring LLMs."""
from typing import Any, Iterator, Sequence

import numpy as np

from faith._internal.algo.matching import AnswerFormat
from faith._internal.metrics.aggregations import cross_count
from faith._internal.metrics.multilabel import micro_f1_score
from faith._internal.metrics.types import MultiLabelSeq, SingleLabelSeq
from faith._internal.types.validation import assert_same_length


def _accuracy(stats: Iterator[tuple[str | None, str | None, bool]]) -> np.float64:
    """Helper function to compute accuracy."""
    return np.mean([1 if lab == pred and proper else 0 for lab, pred, proper in stats])


def _label_metrics(
    label: MultiLabelSeq | SingleLabelSeq, labelset: frozenset[str] | None
) -> dict[str, Any]:
    """Helper function to compute general metrics about the label."""
    return {"query_count": len(label)} | (
        {
            "label_count": {
                lab: np.sum([1 if val == lab else 0 for val in label])
                for lab in labelset
            }
        }
        if labelset is not None
        else {}
    )


def _formatting_metrics(
    answer_format: Sequence[AnswerFormat],
    correct_prediction: Sequence[bool] | None,
) -> dict[str, dict[str, Any]]:
    """Helper function to compute formatting metrics."""
    assert all(
        isinstance(af, AnswerFormat) for af in answer_format
    ), "All answer formats must be of type AnswerFormat."

    answer_format_strs = [str(af) for af in answer_format]
    return (
        {
            "format_breakdown_count": cross_count(
                answer_format_strs,
                ["correct" if cp else "incorrect" for cp in correct_prediction],
                x_dict=set(str(af) for af in AnswerFormat),
                y_dict={"correct", "incorrect"},
            ),
        }
        if correct_prediction is not None
        else {}
    ) | {
        "format_count": {
            str(af): np.sum([1 if a == af else 0 for a in answer_format])
            for af in list(AnswerFormat)
        },
        "format_rate": {
            str(af): np.mean([1 if a == af else 0 for a in answer_format])
            for af in list(AnswerFormat)
        },
    }


def _per_subject_metrics(
    label: SingleLabelSeq,
    prediction: SingleLabelSeq,
    answer_format: Sequence[AnswerFormat],
    subject: SingleLabelSeq | None,
) -> dict[str, Any]:
    subject_list = subject or []
    unique_subjects = sorted(["" if s is None else s for s in set(subject_list)])
    if len(unique_subjects) <= 1:
        # If there is only one subject, per-subject metrics are not meaningful.
        return {}

    accuracy_per_subject = {
        sub: _accuracy(
            (
                (lab, pred, af == AnswerFormat.PROPER)
                for lab, pred, af, s in zip(
                    label, prediction, answer_format, subject_list
                )
                if s == sub
            )
        )
        for sub in unique_subjects
    }
    lenient_accuracy_per_subject = {
        sub: _accuracy(
            (
                (lab, pred, True)
                for lab, pred, s in zip(label, prediction, subject_list)
                if s == sub
            )
        )
        for sub in unique_subjects
    }

    return {
        "accuracy_per_subject": accuracy_per_subject,
        "lenient_accuracy_per_subject": lenient_accuracy_per_subject,
        "subject_weighted_accuracy": np.mean(list(accuracy_per_subject.values())),
        "subject_weighted_lenient_accuracy": np.mean(
            list(lenient_accuracy_per_subject.values())
        ),
    }


def llm_metadata_metrics(
    num_output_tokens: Sequence[int], max_token_halt: Sequence[bool]
) -> dict[str, Any]:
    """Helper function to compute metadata metrics for LLM responses."""
    assert_same_length(
        num_output_tokens=num_output_tokens, max_token_halt=max_token_halt
    )
    return {
        "mean_output_tokens": np.mean(num_output_tokens),
        "rate_max_token_halt": np.mean(max_token_halt),
    }


def llm_prediction_metrics(
    label: SingleLabelSeq,
    prediction: SingleLabelSeq,
    answer_format: Sequence[AnswerFormat],
    subject: SingleLabelSeq | None,
    labelset: frozenset[str] | None,
) -> dict[str, Any]:
    """Helper function to compute core metrics an llm's sufficient statistics."""
    if subject is None:
        assert_same_length(
            label=label, prediction=prediction, answer_format=answer_format
        )
    else:
        assert_same_length(
            label=label,
            prediction=prediction,
            answer_format=answer_format,
            subject=subject,
        )
    assert all(
        lab is None or isinstance(lab, str) for lab in label
    ), "All labels must be strings."
    assert all(
        pred is None or isinstance(pred, str) for pred in prediction
    ), "All predictions must be strings."

    return (
        _label_metrics(label, labelset)
        | _formatting_metrics(
            answer_format, [lab == pred for lab, pred in zip(label, prediction)]
        )
        | {
            "accuracy": _accuracy(
                (
                    (lab, pred, af == AnswerFormat.PROPER)
                    for lab, pred, af in zip(label, prediction, answer_format)
                )
            ),
            "lenient_accuracy": _accuracy(
                ((lab, pred, True) for lab, pred in zip(label, prediction))
            ),
        }
        | _per_subject_metrics(label, prediction, answer_format, subject)
    )


def llm_multilabel_metrics(
    label: MultiLabelSeq,
    prediction: MultiLabelSeq,
    answer_format: Sequence[AnswerFormat],
) -> dict[str, Any]:
    """Helper function to compute core metrics for multilabel llm's sufficient statistics."""
    assert_same_length(label=label, prediction=prediction, answer_format=answer_format)
    disallowed_labels = [
        lab for lab in label if lab is not None and not isinstance(lab, (list, tuple))
    ]
    assert (
        len(disallowed_labels) == 0
    ), f"All labels must be sequences of strings; found: {disallowed_labels}."
    disallowed_predictions = [
        p for p in prediction if p is not None and not isinstance(p, (list, tuple))
    ]
    assert (
        len(disallowed_predictions) == 0
    ), f"All predictions must be sequences of strings; found: {disallowed_predictions}."

    return (
        _label_metrics(label, labelset=None)
        | _formatting_metrics(
            answer_format,
            [set(lab or []) == set(pred or []) for lab, pred in zip(label, prediction)],
        )
        | {
            "accuracy": np.mean(
                [
                    (
                        1
                        if set(lab or []) == set(pred or [])
                        and af == AnswerFormat.PROPER
                        else 0
                    )
                    for lab, pred, af in zip(label, prediction, answer_format)
                ]
            ),
            "micro_f1": micro_f1_score(label, prediction),
        }
    )


def llm_basic_metrics(
    label: SingleLabelSeq,
    prediction: SingleLabelSeq,
    answer_format: Sequence[AnswerFormat],
) -> dict[str, Any]:
    """Helper function to compute domain-specific metrics from predictions based on the expected label."""
    assert_same_length(label=label, prediction=prediction, answer_format=answer_format)
    assert all(
        lab is None or isinstance(lab, str) for lab in label
    ), "All labels must be strings."
    assert all(
        pred is None or isinstance(pred, str) for pred in prediction
    ), "All predictions must be strings."

    return _label_metrics(label, None) | _formatting_metrics(answer_format, None)
