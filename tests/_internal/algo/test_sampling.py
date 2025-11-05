# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from faith._internal.algo.sampling import NShotSampler, sample_partition
from faith._internal.types.flags import SampleRatio


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_sample_partition(n: int) -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
    rng = np.random.default_rng(seed=53124)

    sampled_df, remaining_df = sample_partition(df, n, rng)

    # Check that the partitioned DataFrames have the expected properties.
    assert len(sampled_df) == n, f"Sampled DataFrame should have {n} rows."
    assert len(remaining_df) + len(sampled_df) == len(
        df
    ), "Total number of rows should remain the same."
    assert all(
        row in df.values for row in sampled_df.values
    ), "Sampled rows should be in the original DataFrame."
    assert all(
        row not in sampled_df.values for row in remaining_df.values
    ), "Remaining rows should not be in the sampled DataFrame."

    # Check that the original DataFrame is unchanged.
    assert df.equals(pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]}))


# Test that the sample_partition function is stable for different sample sizes.
@pytest.mark.slow
@pytest.mark.parametrize("df_rows", list(range(10, 101, 30)))
def test_sample_partition_stability(df_rows: int) -> None:
    n_cols = 5
    df = pd.DataFrame(
        np.random.randint(0, 1_000_000, size=(df_rows, n_cols)),
        columns=[f"Col_{i + 1}" for i in range(n_cols)],
    )

    prev_sample: pd.DataFrame | None = None
    for n in range(1, df_rows + 1):
        rng = np.random.default_rng(seed=53124)
        sampled_df, _ = sample_partition(df, n, rng)
        if prev_sample is not None:
            assert prev_sample.equals(
                sampled_df.iloc[:-1]
            ), f"Sampled at size {n} did not incrementally grow"
        prev_sample = sampled_df


def test_sample_partition_invalid_sample_size() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
    rng = np.random.default_rng(seed=42)

    with pytest.raises(ValueError):
        sample_partition(df, len(df) + 1, rng)


def test_nshot_sampler() -> None:
    nshot_sampler = NShotSampler(None, SampleRatio(0), np.random.default_rng(seed=521))
    assert nshot_sampler.get_nshot_examples() is None

    df = pd.DataFrame({"a": [1], "b": ["x"]})
    nshot_sampler = NShotSampler(df, SampleRatio(1), np.random.default_rng(seed=521))
    sample = nshot_sampler.get_nshot_examples()
    assert sample is not None and sample.equals(df)

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    nshot_sampler = NShotSampler(df, SampleRatio(3), np.random.default_rng(seed=521))
    sample = nshot_sampler.get_nshot_examples()
    assert sample is not None and sample.equals(df)

    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": ["x", "y", "z", "w", "v", "u"]})
    nshot_sampler = NShotSampler(df, SampleRatio(3, 6), np.random.default_rng(seed=521))
    sample1 = nshot_sampler.get_nshot_examples()
    assert sample1 is not None and len(sample1) == 3
    sample2 = nshot_sampler.get_nshot_examples()
    assert sample2 is not None and not sample1.equals(sample2)
