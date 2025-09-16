# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Set of functions for aggregating statistics over a set of trials of a benchmark."""
from collections import defaultdict
from numbers import Number
from typing import Any, Sequence, TypeAlias, TypeVar, cast

import numpy as np

# This TypeVar is needed to allow subclasses of Number to be captured.
_NUMBER_TYPE = TypeVar("_NUMBER_TYPE", bound=Number)
BreakdownDict: TypeAlias = dict[str, "BreakdownDict" | _NUMBER_TYPE]


def is_breakdown_dict(obj: Any) -> bool:
    """Checks if an object conforms to the BreakdownDict TypeAlias."""
    return isinstance(obj, dict) and all(
        isinstance(k, str) and (is_breakdown_dict(v) or isinstance(v, Number))
        for k, v in obj.items()
    )


def _raise_inconsistent_types() -> float:
    raise ValueError("Cannot aggregate counts for non-numeric or non-dict values.")


def agg_breakdown_counts(bds: Sequence[BreakdownDict], factor: float) -> BreakdownDict:
    """Aggregate the counts for a given metric across all trials."""
    return {
        k: (
            agg_breakdown_counts(vals, factor)
            if all(isinstance(val, dict) for val in vals)
            else (
                sum(cast(list[float], vals)) * factor
                if all(isinstance(val, Number) or np.isscalar(val) for val in vals)
                else _raise_inconsistent_types()
            )
        )
        for k in set().union(*[bd.keys() for bd in bds])
        if len(vals := [bd[k] for bd in bds if k in bd]) > 0
    }


def agg_trial_stats(
    stats_per_trial: Sequence[Number | BreakdownDict],
) -> BreakdownDict | float:
    """Aggregate the statistics for a given metric across all trials."""
    if len(stats_per_trial) == 0:
        return float("nan")
    if all(isinstance(x, Number) or np.isscalar(x) for x in stats_per_trial):
        return _agg_stats(cast(list[Number], stats_per_trial))
    if all(is_breakdown_dict(x) for x in stats_per_trial):
        dstats = cast(list[BreakdownDict], stats_per_trial)
        return {
            k: agg_trial_stats([x[k] for x in dstats if k in x])
            for k in set().union(*[x.keys() for x in dstats])
        }
    return _raise_inconsistent_types()


def _agg_stats(stats_list: Sequence[Number]) -> BreakdownDict:
    """Compute aggregate statistics for a list of numbers."""
    stats = np.array(stats_list)
    return {
        "mean": float(np.mean(stats)),
        "std": float(np.std(stats)),
        "min": float(np.min(stats)),
        "p_25": float(np.percentile(stats, 25)),
        "median": float(np.median(stats)),
        "p_75": float(np.percentile(stats, 75)),
        "max": float(np.max(stats)),
    }


def cross_count(
    xs: Sequence[str],
    ys: Sequence[str],
    x_dict: set[str],
    y_dict: set[str],
) -> dict[str, dict[str, int]]:
    """Counts the occurrences of each combination of x and y in xs and ys."""
    assert set(xs) <= x_dict, "Items in `xs` must be a subset of `x_dict`"
    assert set(ys) <= y_dict, "Items in `ys` must be a subset of `y_dict`"
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for x, y in zip(xs, ys):
        counts[x][y] += 1
    return {str(x): {str(y): counts[x][y] for y in y_dict} for x in x_dict}
