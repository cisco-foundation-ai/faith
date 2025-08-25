# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for file IO with pandas objects."""
from pathlib import Path

import pandas as pd


def safe_df_to_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Save the `df` to a CSV file at `filepath` ensuring the directory exists.

    Args:
        df: The DataFrame to save.
        filepath: The path to the output directory.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
