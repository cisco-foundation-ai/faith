# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from faith._internal.io.pandas import safe_df_to_csv


def test_safe_df_to_csv() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "new" / "summary.csv"
        df = pd.DataFrame({"col1": [0, 1, 2], "col2": ["a", "b", "c"]})

        # Save the df and check if it is saved correctly.
        safe_df_to_csv(df, filepath)
        assert_frame_equal(pd.read_csv(filepath), df, check_dtype=True)
