# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

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


class TestResolveBigQueryConfig:
    """Tests for BigQuery configuration resolution."""

    def test_resolve_from_flags(self):
        """Test config resolution from explicit flags."""
        project, dataset, table = _resolve_bigquery_config(
            "my-project", "my-dataset", "my-table"
        )
        assert project == "my-project"
        assert dataset == "my-dataset"
        assert table == "my-table"

    def test_resolve_from_env_vars(self):
        """Test config resolution from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "FAITH_BIGQUERY_PROJECT": "env-project",
                "FAITH_BIGQUERY_DATASET": "env-dataset",
                "FAITH_BIGQUERY_TABLE": "env-table",
            },
        ):
            project, dataset, table = _resolve_bigquery_config(None, None, None)
            assert project == "env-project"
            assert dataset == "env-dataset"
            assert table == "env-table"

    def test_resolve_flags_override_env(self):
        """Test that explicit flags override environment variables."""
        with patch.dict(
            "os.environ",
            {
                "FAITH_BIGQUERY_PROJECT": "env-project",
                "FAITH_BIGQUERY_DATASET": "env-dataset",
            },
        ):
            project, dataset, table = _resolve_bigquery_config(
                "flag-project", "flag-dataset", "flag-table"
            )
            assert project == "flag-project"
            assert dataset == "flag-dataset"
            assert table == "flag-table"

    def test_resolve_missing_project(self):
        """Test that missing project raises assertion."""
        with pytest.raises(AssertionError, match="project not specified"):
            _resolve_bigquery_config(None, "dataset", "table")

    def test_resolve_missing_dataset(self):
        """Test that missing dataset raises assertion."""
        with pytest.raises(AssertionError, match="dataset not specified"):
            _resolve_bigquery_config("project", None, "table")

    def test_resolve_default_table(self):
        """Test default table name when not specified."""
        with patch.dict(
            "os.environ",
            {"FAITH_BIGQUERY_PROJECT": "project", "FAITH_BIGQUERY_DATASET": "dataset"},
        ):
            _, _, table = _resolve_bigquery_config(None, None, "metrics")
            assert table == "metrics"


class TestFindMetricsFiles:
    """Tests for finding metrics files."""

    def test_find_metrics_files(self, tmp_path):
        """Test finding metrics.json files recursively."""
        # Create test structure
        (tmp_path / "bench1" / "model1").mkdir(parents=True)
        (tmp_path / "bench1" / "model1" / "metrics.json").write_text("{}")
        (tmp_path / "bench2" / "model2").mkdir(parents=True)
        (tmp_path / "bench2" / "model2" / "metrics.json").write_text("{}")

        metrics_files = _find_metrics_files(tmp_path)
        assert len(metrics_files) == 2
        assert all(f.name == "metrics.json" for f in metrics_files)

    def test_find_metrics_files_not_found(self, tmp_path):
        """Test error when no metrics.json files found."""
        with pytest.raises(FileNotFoundError, match="No metrics.json files found"):
            _find_metrics_files(tmp_path)


class TestProcessMetricsFile:
    """Tests for processing individual metrics files."""

    def test_process_metrics_file_missing_experiment(self, tmp_path):
        """Test that missing experiment.json returns empty list with warning."""
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text('{"stats": {"accuracy": {"mean": 0.8}}}')

        result = _process_metrics_file(metrics_file)
        assert result == []

    def test_process_metrics_file_success(self, tmp_path):
        """Test successful metrics file processing."""
        # Create experiment.json
        exp_file = tmp_path / "experiment.json"
        exp_data = {
            "experiment_params": {
                "benchmark": {
                    "name": "test-bench",
                    "generation_mode": "chat_comp",
                    "prompt_format": "chat",
                    "n_shot": "0",
                },
                "model": {"path": "test/model"},
            },
            "metadata": {},
        }
        exp_file.write_text(json.dumps(exp_data))

        # Create metrics.json
        metrics_file = tmp_path / "metrics.json"
        metrics_data = {"stats": {"accuracy": {"mean": 0.85}}}
        metrics_file.write_text(json.dumps(metrics_data))

        result = _process_metrics_file(metrics_file)
        assert len(result) == 1
        assert result[0]["metric_name"] == "accuracy.mean"
        assert result[0]["metric_value"] == 0.85
        assert result[0]["benchmark"] == "test-bench"
