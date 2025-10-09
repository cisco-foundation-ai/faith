# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Implementation of the subcommand `summarize` to summarize experiment metrics."""

from pathlib import Path
from typing import Sequence

from faith._internal.io.pandas import safe_df_to_csv
from faith.experiment.summarize import build_summary


def summarize_experiments(
    experiment_path: Path, selected_stats: Sequence[str], summary_filepath: Path | None
) -> None:
    """Summarize the experiments in the given path."""
    if summary_filepath is not None and summary_filepath.exists():
        raise FileExistsError(f"Output path {summary_filepath} already exists")

    summary = build_summary(experiment_path, selected_stats)
    if summary_filepath is not None:
        safe_df_to_csv(summary, summary_filepath)
    else:
        print(summary.to_markdown(index=False))
