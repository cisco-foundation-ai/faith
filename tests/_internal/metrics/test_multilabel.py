# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import numpy as np
import pytest

from faith._internal.metrics.multilabel import _tags_to_matrix, micro_f1_score
from faith._internal.metrics.types import MultiLabelSeq


@pytest.mark.parametrize(
    "tags, tag_to_idx, expected_vector",
    [
        (
            ["tag1", "tag2"],
            {"tag1": 0, "tag2": 2, "tag3": 1},
            np.array([1, 0, 1], dtype=int),
        ),
        (["tag3"], {"tag1": 0, "tag2": 1, "tag3": 2}, np.array([0, 0, 1], dtype=int)),
        ([], {"tag1": 0, "tag2": 1, "tag3": 2}, np.array([0, 0, 0], dtype=int)),
    ],
)
def test_tags_to_matrix(
    tags: Sequence[str] | None, tag_to_idx: dict[str, int], expected_vector: np.ndarray
) -> None:
    result = _tags_to_matrix(tags, tag_to_idx)
    np.testing.assert_array_equal(result, expected_vector)


@pytest.mark.parametrize(
    "label, prediction, expected_score",
    [
        ([], [], float("nan")),
        ([["tag1"]], [["tag2", "tag3"]], 0.0),
        ([["tag1", "tag2"]], [["tag2", "tag1"]], 1.0),
        ([["tag1", "tag2"]], [["tag2", "tag3"]], 0.5),
        ([[], ["tag1"]], [[], ["tag1"]], 1.0),
        (
            [["tag1", "tag2"], ["tag2"], ["tag3"]],
            [["tag2", "tag1"], ["tag2", "tag2"], ["tag3"]],
            1.0,
        ),
        (
            [["tag1", "tag2"], ["tag2"], ["tag3"]],
            [["tag1"], ["tag2", "tag3"], ["tag3"]],
            0.75,
        ),
        ([["tag1"], ["tag1"]], [None, ["tag1"]], 0.5),
        ([None, ["tag1"]], [["tag1"], ["tag1"]], 0.5),
        ([None, ["tag1"]], [None, ["tag1"]], 1.0),
    ],
)
def test_micro_f1_score(
    label: MultiLabelSeq, prediction: MultiLabelSeq, expected_score: float
) -> None:
    score = micro_f1_score(label, prediction)
    if np.isnan(expected_score):
        assert np.isnan(score), f"Expected NaN, got {score}"
    else:
        assert np.isclose(
            score, expected_score
        ), f"Expected {expected_score}, got {score}"


def test_micro_f1_score_unbalanced() -> None:
    with pytest.raises(AssertionError):
        micro_f1_score([["tag1"]], [["tag2", "tag3", "tag4"], ["tag5"]])
