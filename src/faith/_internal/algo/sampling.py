# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for sampling data from DataFrames."""

from typing import TypeVar

import numpy as np
import pandas as pd

from faith._types.benchmark.sample_ratio import SampleRatio

T = TypeVar("T")


def sample_sequence(items: list[T], n: int, rng: np.random.Generator) -> list[T]:
    """Sample `n` items from `items` without replacement, preserving their order.

    The list-level analog of `sample_partition`. Sampling is stable across sizes:
    for the same `items` and `rng`, the size-`n` result is an order-preserving
    subset of the size-`n+1` result. If `n >= len(items)`, the list is returned
    unchanged.

    Args:
        items: The list to sample from (not modified).
        n: The number of items to sample.
        rng: Random number generator.

    Returns:
        The sampled items in their original relative order.

    Raises:
        ValueError: If `n` is negative.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    if n >= len(items):
        return items
    sampled_positions = sorted(rng.permutation(len(items))[:n])
    return [items[i] for i in sampled_positions]


def sample_partition(
    df: pd.DataFrame, n: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample `n` rows from `df` without replacement and return (sample, remainder).

    This function partitions `df` into two parts by selecting `n` rows randomly.
    The first part contains the sampled rows, and the second is the remaining rows.
    Note: This sampling is done without replacement and is stable for different
    sample sizes `n`; ie. for the same `df` and `rng`, calling `sample_partition`
    for `n` and `n+1` will yield a sampled DataFrame in the first `n` rows of both.
    Note: The input DataFrame `df` is not modified by this function.

    Args:
        df (pd.DataFrame): The DataFrame to sample from.
        n (int): The number of rows to sample.
        rng (np.random.Generator, optional): Random number generator.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the sampled DataFrame
            and the remaining DataFrame.

    Raises:
        ValueError: If `n` is greater than the number of rows in `df`.
    """
    if n > len(df):
        raise ValueError("Sample size cannot be larger than the DataFrame size.")
    random_perm = rng.permutation(len(df))
    sample_indices = random_perm[:n]
    sampled_df = df.iloc[sample_indices]
    remaining_df = df.drop(sampled_df.index)

    return sampled_df, remaining_df


class NShotSampler:
    """Container for constructing n-shot examples."""

    def __init__(
        self,
        nshot_data: pd.DataFrame | None,
        n_shot: SampleRatio,
        rng: np.random.Generator,
    ):
        """Initialize the n-shot examples container."""
        num_samples = (
            n_shot.numerator if n_shot.denominator == 1 else n_shot.denominator
        )
        assert (
            num_samples == 0 or nshot_data is not None
        ), "nshot_data must be provided for n-shot examples."
        assert (
            nshot_data is None or len(nshot_data) == num_samples
        ), "nshot_data must have the same number of examples as n_shot."
        self._nshot_data = nshot_data
        self._n_shot = n_shot
        self._rng = rng

    def get_nshot_examples(self) -> pd.DataFrame | None:
        """Get the n-shot examples for the prompt."""
        if self._nshot_data is None:
            return None
        if self._n_shot.denominator > 1:
            return self._nshot_data.sample(
                self._n_shot.numerator, random_state=self._rng, replace=False
            )
        return self._nshot_data
