# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from faith._internal.algo.matching import AnswerFormat
from faith._internal.metrics.llm import (
    llm_basic_metrics,
    llm_judge_grades,
    llm_metadata_metrics,
    llm_multilabel_metrics,
    llm_prediction_metrics,
)


def test_llm_metadata_metrics() -> None:
    num_output_tokens = [0, 4, 7, 8, 2, 3]
    max_token_halt = [False, True, False, False, True, False]

    assert llm_metadata_metrics(num_output_tokens, max_token_halt) == {
        "mean_output_tokens": pytest.approx(4.0),
        "rate_max_token_halt": pytest.approx(1 / 3),
    }


def test_llm_prediction_metrics() -> None:
    label = ["answer1", "answer2", "answer3", "answer4"]
    prediction = ["answer1", "wrong_answer", "answer3", None]
    answer_format = [
        AnswerFormat.PROPER,
        AnswerFormat.PROPER,
        AnswerFormat.IMPROPER,
        AnswerFormat.INVALID,
    ]
    subject = ["dysania", "aglets", "aglets", "aglets"]

    assert llm_prediction_metrics(
        label, prediction, answer_format, subject, labelset=None
    ) == {
        "query_count": 4,
        "accuracy": pytest.approx(1 / 4),
        "lenient_accuracy": pytest.approx(1 / 2),
        "format_breakdown_count": {
            "proper": {"correct": 1, "incorrect": 1},
            "improper": {"correct": 1, "incorrect": 0},
            "invalid": {"correct": 0, "incorrect": 1},
            "inferred": {"correct": 0, "incorrect": 0},
        },
        "format_count": {
            "proper": 2,
            "improper": 1,
            "invalid": 1,
            "inferred": 0,
        },
        "format_rate": {
            "proper": pytest.approx(1 / 2),
            "improper": pytest.approx(1 / 4),
            "invalid": pytest.approx(1 / 4),
            "inferred": 0.0,
        },
        "accuracy_per_subject": {
            "dysania": pytest.approx(1),
            "aglets": pytest.approx(0),
        },
        "lenient_accuracy_per_subject": {
            "dysania": pytest.approx(1),
            "aglets": pytest.approx(1 / 3),
        },
        "subject_weighted_accuracy": pytest.approx(0.5),
        "subject_weighted_lenient_accuracy": pytest.approx(2 / 3),
    }

    assert llm_prediction_metrics(
        label,
        prediction,
        answer_format,
        None,
        labelset=frozenset({"answer1", "answer2", "answer3", "answer4"}),
    ) == {
        "query_count": 4,
        "accuracy": pytest.approx(1 / 4),
        "lenient_accuracy": pytest.approx(1 / 2),
        "format_breakdown_count": {
            "proper": {"correct": 1, "incorrect": 1},
            "improper": {"correct": 1, "incorrect": 0},
            "invalid": {"correct": 0, "incorrect": 1},
            "inferred": {"correct": 0, "incorrect": 0},
        },
        "format_count": {
            "proper": 2,
            "improper": 1,
            "invalid": 1,
            "inferred": 0,
        },
        "format_rate": {
            "proper": pytest.approx(1 / 2),
            "improper": pytest.approx(1 / 4),
            "invalid": pytest.approx(1 / 4),
            "inferred": 0.0,
        },
        "label_count": {"answer1": 1, "answer2": 1, "answer3": 1, "answer4": 1},
    }


def test_llm_multilabel_metrics() -> None:
    label = [["tag1", "tag2"], ["tag2"], ["tag3"], ["tag2"]]
    prediction = [["tag1", "tag2"], ["tag2", "tag3"], ["tag3"], None]
    answer_format = [
        AnswerFormat.PROPER,
        AnswerFormat.IMPROPER,
        AnswerFormat.PROPER,
        AnswerFormat.INVALID,
    ]

    assert llm_multilabel_metrics(label, prediction, answer_format) == {
        "query_count": 4,
        "accuracy": pytest.approx(2 / 4),
        "micro_f1": pytest.approx(4 / 5),
        "format_breakdown_count": {
            "proper": {"correct": 2, "incorrect": 0},
            "improper": {"correct": 0, "incorrect": 1},
            "invalid": {"correct": 0, "incorrect": 1},
            "inferred": {"correct": 0, "incorrect": 0},
        },
        "format_count": {
            "proper": 2,
            "improper": 1,
            "invalid": 1,
            "inferred": 0,
        },
        "format_rate": {
            "proper": pytest.approx(1 / 2),
            "improper": pytest.approx(1 / 4),
            "invalid": pytest.approx(1 / 4),
            "inferred": 0.0,
        },
    }


def test_llm_judge_grades() -> None:
    label = ["a b c", "a f d c", "a a", "b b c"]
    prediction = ["a b c", "a e d", "a b", "a b c"]
    answer_format = [
        AnswerFormat.PROPER,
        AnswerFormat.PROPER,
        AnswerFormat.IMPROPER,
        AnswerFormat.INVALID,
    ]
    judgements = [
        {"awarded_points": 3.0, "min_points": 0.0, "max_points": 3.0},
        {"awarded_points": 2.0, "min_points": 0.0, "max_points": 4.0},
        {"awarded_points": 1.0, "min_points": 0.0, "max_points": 2.0},
        {"awarded_points": 2.0, "min_points": 0.0, "max_points": 3.0},
    ]
    assert llm_judge_grades(label, prediction, answer_format, judgements) == {
        "query_count": 4,
        "format_count": {
            "proper": 2,
            "improper": 1,
            "invalid": 1,
            "inferred": 0,
        },
        "format_rate": {
            "proper": pytest.approx(1 / 2),
            "improper": pytest.approx(1 / 4),
            "invalid": pytest.approx(1 / 4),
            "inferred": 0.0,
        },
        "mean_grade": pytest.approx(2 / 3),
        "median_grade": pytest.approx(7 / 12),
        "stddev_grade": pytest.approx(1 / 2 / math.sqrt(6)),
    }


def test_llm_basic_metrics() -> None:
    label = ["foo", "bar", "baz"]
    prediction = ["foo", "faz", "qux"]
    answer_format = [AnswerFormat.PROPER, AnswerFormat.IMPROPER, AnswerFormat.PROPER]

    assert llm_basic_metrics(label, prediction, answer_format) == {
        "query_count": 3,
        "format_count": {
            "proper": 2,
            "improper": 1,
            "invalid": 0,
            "inferred": 0,
        },
        "format_rate": {
            "proper": pytest.approx(2 / 3),
            "improper": pytest.approx(1 / 3),
            "invalid": 0.0,
            "inferred": 0.0,
        },
    }
