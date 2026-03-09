# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import tempfile
from pathlib import Path
from unittest.mock import ANY, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from faith.cli.subcmd.summarize import (
    _find_metrics_files,
    _process_metrics_file,
    _resolve_bigquery_config,
    summarize_experiments,
)


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


def test_resolve_from_flags():
    """Test config resolution from explicit flags."""
    assert _resolve_bigquery_config("my-project", "my-dataset", "my-table") == (
        "my-project",
        "my-dataset",
        "my-table",
    )


def test_resolve_from_env_vars():
    """Test config resolution from environment variables."""
    with patch.dict(
        "os.environ",
        {
            "FAITH_BIGQUERY_PROJECT": "env-project",
            "FAITH_BIGQUERY_DATASET": "env-dataset",
            "FAITH_BIGQUERY_TABLE": "env-table",
        },
    ):
        assert _resolve_bigquery_config(None, None, None) == (
            "env-project",
            "env-dataset",
            "env-table",
        )


def test_resolve_flags_override_env():
    """Test that explicit flags override environment variables."""
    with patch.dict(
        "os.environ",
        {
            "FAITH_BIGQUERY_PROJECT": "env-project",
            "FAITH_BIGQUERY_DATASET": "env-dataset",
        },
    ):
        assert _resolve_bigquery_config(
            "flag-project", "flag-dataset", "flag-table"
        ) == (
            "flag-project",
            "flag-dataset",
            "flag-table",
        )


def test_resolve_missing_project():
    """Test that missing project raises assertion."""
    with pytest.raises(AssertionError, match="project not specified"):
        _resolve_bigquery_config(None, "dataset", "table")


def test_resolve_missing_dataset():
    """Test that missing dataset raises assertion."""
    with pytest.raises(AssertionError, match="dataset not specified"):
        _resolve_bigquery_config("project", None, "table")


def test_resolve_default_table():
    """Test default table name when not specified."""
    with patch.dict(
        "os.environ",
        {"FAITH_BIGQUERY_PROJECT": "project", "FAITH_BIGQUERY_DATASET": "dataset"},
    ):
        _, _, table = _resolve_bigquery_config(None, None, "metrics")
        assert table == "metrics"


def test_find_metrics_files(tmp_path):
    """Test finding metrics.json files recursively."""
    (tmp_path / "bench1" / "model1").mkdir(parents=True)
    (tmp_path / "bench1" / "model1" / "metrics.json").write_text("{}")
    (tmp_path / "bench2" / "model2").mkdir(parents=True)
    (tmp_path / "bench2" / "model2" / "metrics.json").write_text("{}")

    metrics_files = _find_metrics_files(tmp_path)
    assert len(metrics_files) == 2
    assert all(f.name == "metrics.json" for f in metrics_files)


def test_find_metrics_files_not_found(tmp_path):
    """Test error when no metrics.json files found."""
    with pytest.raises(FileNotFoundError, match="No metrics.json files found"):
        _find_metrics_files(tmp_path)


def test_process_metrics_file_missing_experiment():
    """Test that missing experiment.json raises error."""
    testdata_dir = Path(__file__).parent / "testdata" / "missing_experiment"
    with pytest.raises(FileNotFoundError, match="Missing experiment.json"):
        _process_metrics_file(testdata_dir / "metrics.json")


def test_process_metrics_file_success():
    """Test successful metrics file processing."""
    testdata_dir = Path(__file__).parent / "testdata" / "process_metrics"
    result = _process_metrics_file(testdata_dir / "metrics.json")
    assert result == [
        {
            "metric_name": "accuracy.mean",
            "metric_value": 0.85,
            "is_primary": True,
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "test/model",
            "source_uri": "test/model",
            "benchmark": "test-bench",
            "temperature": None,
            "top_p": None,
            "max_completion_tokens": None,
            "context_length": None,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_shots": 0,
            "num_shots_pool_size": 1,
        }
    ]
