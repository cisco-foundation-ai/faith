# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from numbers import Number
from typing import Sequence

import pytest

from faith._internal.metrics.aggregations import (
    BreakdownDict,
    agg_trial_stats,
    cross_count,
)


@pytest.mark.parametrize(
    "trial_stats, expected_breakdown",
    [
        (
            [],
            pytest.approx(float("nan"), nan_ok=True),
        ),
        (
            [1, 3],
            {
                "mean": 2.0,
                "std": 1.0,
                "min": 1,
                "p_25": 1.5,
                "median": 2.0,
                "p_75": 2.5,
                "max": 3,
            },
        ),
        (
            [{"a": 1, "b": 7}, {"a": 3, "b": 5}, {"a": 2, "b": 6}],
            {
                "a": {
                    "mean": 2,
                    "std": pytest.approx(0.816496580927726),
                    "min": 1,
                    "p_25": pytest.approx(3 / 2),
                    "median": 2,
                    "p_75": pytest.approx(5 / 2),
                    "max": 3,
                },
                "b": {
                    "mean": 6,
                    "std": pytest.approx(0.816496580927726),
                    "min": 5,
                    "p_25": pytest.approx(11 / 2),
                    "median": 6,
                    "p_75": pytest.approx(13 / 2),
                    "max": 7,
                },
            },
        ),
    ],
)
def test_agg_trial_stats(
    trial_stats: Sequence[Number] | Sequence[BreakdownDict],
    expected_breakdown: BreakdownDict,
) -> None:
    assert agg_trial_stats(trial_stats) == expected_breakdown


@pytest.mark.parametrize(
    "xs, ys, x_dict, y_dict, expected",
    [
        (
            ["a", "b", "a"],
            ["x", "y", "x"],
            {"a", "b"},
            {"x", "y"},
            {"a": {"x": 2, "y": 0}, "b": {"x": 0, "y": 1}},
        ),
        (
            ["a", "b", "b"],
            ["x", "x", "x"],
            {"a", "b"},
            {"x", "y"},
            {"a": {"x": 1, "y": 0}, "b": {"x": 2, "y": 0}},
        ),
        (
            ["b", "b", "b"],
            ["y", "y", "x"],
            {"a", "b"},
            {"x", "y"},
            {"a": {"x": 0, "y": 0}, "b": {"x": 1, "y": 2}},
        ),
        ([], [], {"a"}, {"x"}, {"a": {"x": 0}}),
    ],
)
def test_cross_count(
    xs: Sequence[str],
    ys: Sequence[str],
    x_dict: set[str],
    y_dict: set[str],
    expected: dict[str, dict[str, int]],
) -> None:
    result = cross_count(xs, ys, x_dict, y_dict)
    assert result == expected


def test_cross_count_invalid_x_vocab() -> None:
    with pytest.raises(AssertionError):
        cross_count(["a", "b", "a"], ["x", "y", "x"], {"a"}, {"x", "y"})

    with pytest.raises(AssertionError):
        cross_count(["a", "b", "a"], ["x", "y", "x"], {"a", "b"}, {"x"})
