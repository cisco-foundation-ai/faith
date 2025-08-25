# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from faith.experiment.summarize import build_summary


def test_build_summary() -> None:
    experiment_path = Path(__file__).parent / "testdata"
    selected_stats = ["accuracy"]

    # Build the summary
    summary = build_summary(experiment_path, selected_stats)
    assert_frame_equal(
        summary,
        pd.DataFrame(
            {
                "benchmark": ["ctibench-rcm", "ctibench-rcm"],
                "model": ["syntax-and-sensibility-3B", "wordy-mc-worder-27B"],
                "prompt_format": ["chat", "chat"],
                "gen_mode": ["chat_comp", "chat_comp"],
                "n_shot": ["0", "0"],
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
