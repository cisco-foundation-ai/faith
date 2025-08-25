# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Library of functions for summarizing the metrics from benchmark experiments."""
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from faith._internal.io.json import read_json_file


def _build_experiment_record(
    exp_path: Path, metric_path: Path, selected_stats: Sequence[str]
) -> dict[str, Any]:
    """Build a record for a single benchmark experiment."""
    assert (
        exp_path.parent == metric_path.parent
    ), f"Experiment path '{exp_path}' and metric path '{metric_path}' do not match."

    experiment = read_json_file(exp_path)
    metrics = read_json_file(metric_path)

    return {
        "benchmark": experiment["experiment_params"]["benchmark"]["name"],
        "model": experiment["experiment_params"]["model"]["name"],
        "prompt_format": experiment["experiment_params"]["benchmark"]["prompt_format"],
        "gen_mode": experiment["experiment_params"]["benchmark"]["generation_mode"],
        "n_shot": experiment["experiment_params"]["benchmark"]["n_shot"],
    } | {
        f"{stat}_{substat}": value
        for stat in selected_stats
        for substat, value in metrics["stats"].get(stat, {}).items()
    }


def build_summary(experiment_path: Path, selected_stats: Sequence[str]) -> pd.DataFrame:
    """Build a summary of an experiment's metrics.

    Args:
        experiment_path: The root path to the experiment to summarize.
        selected_stats: A list of statistics to include in the summary.

    Returns:
        A dictionary containing the summary of the experiment's metrics.
    """
    assert (
        experiment_path.exists() and experiment_path.is_dir()
    ), "The experiment path must be a directory."

    metric_paths = list(experiment_path.glob("**/metrics.json"))
    experiment_paths = [path.parent / "experiment.json" for path in metric_paths]
    assert all(
        path.exists() for path in experiment_paths
    ), f"There must be an 'experiment.json' file for each 'metrics.json'; missing: {[path for path in experiment_paths if not path.exists()]}"

    table = pd.DataFrame(
        _build_experiment_record(ep, mp, selected_stats)
        for ep, mp in zip(experiment_paths, metric_paths)
    )
    table["n_shot"] = table["n_shot"].astype(str)
    return table.sort_values(by=["benchmark", "model"]).reset_index(drop=True)
