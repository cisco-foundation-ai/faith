# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import tempfile
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from faith.cli.subcmd.summarize import summarize_experiments


def test_summarize_experiments() -> None:
    experiment_path = Path(__file__).parent / "testdata" / "summarize"
    selected_stats = ["accuracy"]

    with tempfile.TemporaryDirectory() as temp_dir:
        summary_filepath = Path(temp_dir) / "new" / "summary.csv"
        summarize_experiments(experiment_path, selected_stats, summary_filepath)
        assert_frame_equal(
            pd.read_csv(summary_filepath),
            pd.DataFrame(
                {
                    "benchmark": ["ctibench-rcm", "ctibench-rcm"],
                    "model": ["syntax-and-sensibility-3B", "wordy-mc-worder-27B"],
                    "prompt_format": ["chat", "chat"],
                    "gen_mode": ["chat_comp", "chat_comp"],
                    "n_shot": ["0", "1/2"],
                    "temp": [0.7, 0.7],
                    "accuracy_max": [0.176, 0.569],
                    "accuracy_mean": [0.1649, 0.5581],
                    "accuracy_median": [0.162, 0.558],
                    "accuracy_min": [0.148, 0.544],
                    "accuracy_p_25": [0.16025, 0.55625],
                    "accuracy_p_75": [0.1735, 0.564],
                    "accuracy_std": [0.0087801, 0.0072863],
                }
            ),
            check_dtype=True,
        )

    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        summarize_experiments(experiment_path, selected_stats, None)
        assert (
            buf.getvalue()
            == """| benchmark    | model                     | prompt_format   | gen_mode   | n_shot   |   temp |   accuracy_max |   accuracy_mean |   accuracy_median |   accuracy_min |   accuracy_p_25 |   accuracy_p_75 |   accuracy_std |
|:-------------|:--------------------------|:----------------|:-----------|:---------|-------:|---------------:|----------------:|------------------:|---------------:|----------------:|----------------:|---------------:|
| ctibench-rcm | syntax-and-sensibility-3B | chat            | chat_comp  | 0        |    0.7 |          0.176 |          0.1649 |             0.162 |          0.148 |         0.16025 |          0.1735 |      0.0087801 |
| ctibench-rcm | wordy-mc-worder-27B       | chat            | chat_comp  | 1/2      |    0.7 |          0.569 |          0.5581 |             0.558 |          0.544 |         0.55625 |          0.564  |      0.0072863 |
"""
        )


def test_summarize_experiments_fails_on_existing() -> None:
    experiment_path = Path(__file__).parent / "testdata" / "summarize"
    selected_stats = ["accuracy"]

    with tempfile.TemporaryDirectory() as temp_dir:
        summary_filepath = Path(temp_dir) / "new" / "summary.csv"
        summarize_experiments(experiment_path, selected_stats, summary_filepath)

        # Attempting to summarize again should raise FileExistsError.
        try:
            summarize_experiments(experiment_path, selected_stats, summary_filepath)
        except FileExistsError as e:
            assert str(e) == f"Output path {summary_filepath} already exists"
