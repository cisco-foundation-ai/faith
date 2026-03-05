# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for FAITH metrics parser."""

# pylint: disable=redefined-outer-name  # pytest fixtures

import json

import pytest

from faith.ingestion.parser import (
    ExperimentConfig,
    parse_experiment_config,
    parse_metrics_file,
)


class TestParseExperimentConfig:
    """Tests for parse_experiment_config function."""

    @pytest.fixture
    def experiment_data(self):
        """Sample experiment.json data."""
        return {
            "benchmark_config": {
                "metadata": {
                    "name": "ctibench-mcqa",
                },
                "output_processing": {
                    "primary_metric": "accuracy.mean",
                },
            },
            "experiment_params": {
                "benchmark": {
                    "generation_mode": "chat_comp",
                    "n_shot": "0",
                    "name": "ctibench-mcqa",
                    "prompt_format": "chat",
                },
                "model": {
                    "name": "Llama-3.1-8B-Instruct",
                    "path": "meta-llama/Llama-3.1-8B-Instruct",
                    "engine": {
                        "context_length": 3500,
                        "engine_type": "vllm",
                        "kwargs": {},
                        "num_gpus": 1,
                    },
                    "generation": {
                        "kwargs": {},
                        "max_completion_tokens": 1024,
                        "temperature": 0.3,
                        "top_p": 1.0,
                    },
                },
            },
            "metadata": {
                "run_args": [
                    "/usr/local/bin/faith",
                    "run-all",
                    "--num-trials",
                    "10",
                    "--seed",
                    "42",
                    "--sample-size",
                    "100",
                ],
            },
        }

    def test_parse_config(self, experiment_data):
        """Test parsing experiment config."""
        config = parse_experiment_config(experiment_data)

        assert isinstance(config, ExperimentConfig)
        assert config.model_name == "Llama-3.1-8B-Instruct"
        assert config.model_path == "meta-llama/Llama-3.1-8B-Instruct"
        assert config.benchmark_name == "ctibench-mcqa"
        assert config.temperature == 0.3
        assert config.top_p == 1.0
        assert config.max_completion_tokens == 1024
        assert config.context_length == 3500
        assert config.generation_mode == "chat_comp"
        assert config.prompt_format == "chat"
        assert config.num_shots == 0
        assert config.num_shots_pool_size == 1
        assert config.num_trials == 10
        assert config.seed == 42
        assert config.sample_size == 100
        assert config.primary_metric_name == "accuracy.mean"

    def test_parse_missing_benchmark_name(self):
        """Test parsing config with incomplete benchmark spec raises error."""
        data = {"experiment_params": {"benchmark": {}, "model": {"path": "test/model"}}}

        # BenchmarkSpec.from_dict() will raise KeyError for missing required fields
        with pytest.raises(KeyError):
            parse_experiment_config(data)

    def test_parse_num_shots_string_conversion(self):
        """Test that n_shot string is converted to int."""
        data = {
            "experiment_params": {
                "benchmark": {
                    "name": "test-bench",
                    "generation_mode": "chat_comp",
                    "prompt_format": "chat",
                    "n_shot": "5",
                },
                "model": {"path": "test/model"},
            },
            "metadata": {},
        }

        config = parse_experiment_config(data)
        assert config.num_shots == 5
        assert config.num_shots_pool_size == 1
        assert isinstance(config.num_shots, int)
        # Optional fields should be None when not provided
        assert config.seed is None
        assert config.sample_size is None

    def test_parse_num_shots_fractional(self):
        """Test that fractional n_shot (e.g., '1/2') extracts both fields."""
        data = {
            "experiment_params": {
                "benchmark": {
                    "name": "test-bench",
                    "generation_mode": "chat_comp",
                    "prompt_format": "chat",
                    "n_shot": "3/5",
                },
                "model": {"path": "test/model"},
            },
            "metadata": {},
        }

        config = parse_experiment_config(data)
        assert config.num_shots == 3
        assert config.num_shots_pool_size == 5

    def test_parse_primary_metric_from_config(self):
        """Test reading primary_metric from benchmark config output_processing."""
        data = {
            "benchmark_config": {
                "output_processing": {
                    "primary_metric": "custom_metric.mean",
                }
            },
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

        config = parse_experiment_config(data)
        assert config.primary_metric_name == "custom_metric.mean"

    def test_parse_missing_primary_metric(self):
        """Test parsing without primary_metric returns None."""
        data = {
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

        config = parse_experiment_config(data)
        assert config.primary_metric_name is None


@pytest.fixture
def experiment_config():
    """Sample experiment config."""
    return ExperimentConfig(
        model_name="Llama-3.1-8B-Instruct",
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        benchmark_name="ctibench-mcqa",
        temperature=0.3,
        top_p=1.0,
        max_completion_tokens=1024,
        context_length=3500,
        generation_mode="chat_comp",
        prompt_format="chat",
        num_shots=0,
        num_shots_pool_size=1,
        num_trials=10,
        primary_metric_name="accuracy.mean",
    )


@pytest.fixture
def metrics_data_simple():
    """Simple metrics.json data for testing."""
    return {
        "stats": {
            "accuracy": {
                "mean": 0.8875,
                "std": 0.0096,
                "min": 0.8625,
                "max": 0.9,
            },
            "weighted_avg_f1": {
                "mean": 0.8866,
                "std": 0.01,
            },
        }
    }


@pytest.fixture
def metrics_data_nested():
    """Nested metrics.json data (like f1_scores)."""
    return {
        "stats": {
            "accuracy": {
                "mean": 0.4608,
                "std": 0.0,
            },
            "f1_scores": {
                "A": {"mean": 0.511, "std": 0.0},
                "B": {"mean": 0.650, "std": 0.0},
                "C": {"mean": 0.659, "std": 0.0},
                "D": {"mean": 0.511, "std": 0.0},
            },
        }
    }


class TestParseMetricsBasic:
    """Tests for basic parsing functionality."""

    def test_parse_simple_metrics_common_fields(
        self, experiment_config, metrics_data_simple, tmp_path
    ):
        """Test parsing simple flat metrics - common fields."""
        metrics_file = tmp_path / "gen_params_abc123" / "metrics.json"
        metrics_file.parent.mkdir(parents=True)
        metrics_file.write_text(json.dumps(metrics_data_simple))

        results = parse_metrics_file(
            metrics_file,
            experiment_config,
            model_key="test-model",
            source_uri="gs://bucket/models/test-model",
        )

        # Should have 6 metrics (4 for accuracy, 2 for weighted_avg_f1)
        assert len(results) == 6

        # Check common fields
        for record in results:
            assert "metrics.json" in record["metrics_file_uri"]
            assert record["model_key"] == "test-model"
            assert record["source_uri"] == "gs://bucket/models/test-model"
            assert record["benchmark"] == "ctibench-mcqa"
            assert "faith_version" in record
            assert isinstance(record["faith_version"], str)
            assert len(record["faith_version"]) > 0
            assert record["temperature"] == 0.3
            assert record["top_p"] == 1.0
            assert record["max_completion_tokens"] == 1024
            assert record["context_length"] == 3500
            assert record["num_trials"] == 10
            assert record["num_shots"] == 0
            assert record["num_shots_pool_size"] == 1

    def test_parse_simple_metrics_values(
        self, experiment_config, metrics_data_simple, tmp_path
    ):
        """Test parsing simple flat metrics - metric values and names."""
        metrics_file = tmp_path / "gen_params_abc123" / "metrics.json"
        metrics_file.parent.mkdir(parents=True)
        metrics_file.write_text(json.dumps(metrics_data_simple))

        results = parse_metrics_file(
            metrics_file,
            experiment_config,
            model_key="test-model",
            source_uri="gs://bucket/models/test-model",
        )

        # Check metric names are flattened correctly
        metric_names = [r["metric_name"] for r in results]
        assert "accuracy.mean" in metric_names
        assert "accuracy.std" in metric_names
        assert "accuracy.min" in metric_names
        assert "accuracy.max" in metric_names
        assert "weighted_avg_f1.mean" in metric_names
        assert "weighted_avg_f1.std" in metric_names

        # Check metric values
        accuracy_mean_record = next(
            r for r in results if r["metric_name"] == "accuracy.mean"
        )
        assert accuracy_mean_record["metric_value"] == 0.8875
        assert accuracy_mean_record["metric_unit"] == "ratio"
        assert (
            accuracy_mean_record["is_primary"] is True
        )  # accuracy.mean is primary for ctibench-mcqa

    def test_parse_nested_metrics(
        self, experiment_config, metrics_data_nested, tmp_path
    ):
        """Test parsing nested metrics (like f1_scores)."""
        metrics_file = tmp_path / "gen_params_xyz789" / "metrics.json"
        metrics_file.parent.mkdir(parents=True)
        metrics_file.write_text(json.dumps(metrics_data_nested))

        results = parse_metrics_file(
            metrics_file,
            experiment_config,
            model_key="test-model",
        )

        # Should have 10 metrics: accuracy (2) + f1_scores.A/B/C/D (2 each = 8)
        assert len(results) == 10

        # Check nested metric names are flattened correctly
        metric_names = [r["metric_name"] for r in results]
        assert "accuracy.mean" in metric_names
        assert "accuracy.std" in metric_names
        assert "f1_scores.A.mean" in metric_names
        assert "f1_scores.A.std" in metric_names
        assert "f1_scores.B.mean" in metric_names
        assert "f1_scores.C.mean" in metric_names
        assert "f1_scores.D.mean" in metric_names

        # Check primary metric marking
        accuracy_mean_record = next(
            r for r in results if r["metric_name"] == "accuracy.mean"
        )
        assert accuracy_mean_record["is_primary"] is True

        f1_record = next(r for r in results if r["metric_name"] == "f1_scores.A.mean")
        assert f1_record["is_primary"] is False

    def test_parse_without_gen_params_hash(
        self, experiment_config, metrics_data_simple, tmp_path
    ):
        """Test parsing when path doesn't contain gen_params_hash."""
        metrics_file = tmp_path / "no_hash_directory" / "metrics.json"
        metrics_file.parent.mkdir(parents=True)
        metrics_file.write_text(json.dumps(metrics_data_simple))

        results = parse_metrics_file(metrics_file, experiment_config)

        # Should still work
        assert len(results) > 0
        assert "metrics_file_uri" in results[0]

    def test_parse_invalid_stats_type(self, experiment_config, tmp_path):
        """Test parsing metrics.json with invalid stats type."""
        metrics_file = tmp_path / "gen_params_abc123" / "metrics.json"
        metrics_file.parent.mkdir(parents=True)
        metrics_file.write_text(json.dumps({"stats": "not a dict"}))

        with pytest.raises(AssertionError, match="missing or invalid 'stats' dict"):
            parse_metrics_file(metrics_file, experiment_config)

    def test_parse_nonexistent_file(self, experiment_config, tmp_path):
        """Test parsing nonexistent metrics file."""
        with pytest.raises(FileNotFoundError):
            parse_metrics_file(tmp_path / "nonexistent.json", experiment_config)


class TestParseMetricsAdvanced:
    """Tests for advanced parsing features."""

    def test_num_shots_in_config(
        self, experiment_config, metrics_data_simple, tmp_path
    ):
        """Test that num_shots from config is used."""
        metrics_file = tmp_path / "gen_params_abc123" / "metrics.json"
        metrics_file.parent.mkdir(parents=True)
        metrics_file.write_text(json.dumps(metrics_data_simple))

        results = parse_metrics_file(metrics_file, experiment_config)

        assert results[0]["num_shots"] == 0
        assert results[0]["num_shots_pool_size"] == 1
        assert isinstance(results[0]["num_shots"], int)

    def test_metric_units(self, experiment_config, tmp_path):
        """Test metric unit inference."""
        metrics_data = {
            "stats": {
                "accuracy": {"mean": 0.85},
                "large_value": {"mean": 150.5},
            }
        }

        metrics_file = tmp_path / "gen_params_abc123" / "metrics.json"
        metrics_file.parent.mkdir(parents=True)
        metrics_file.write_text(json.dumps(metrics_data))

        results = parse_metrics_file(metrics_file, experiment_config)

        accuracy_record = next(
            r for r in results if r["metric_name"] == "accuracy.mean"
        )
        assert accuracy_record["metric_unit"] == "ratio"

        large_value_record = next(
            r for r in results if r["metric_name"] == "large_value.mean"
        )
        assert large_value_record["metric_unit"] is None

    def test_optional_fields(self, experiment_config, metrics_data_simple, tmp_path):
        """Test optional model_key and source_uri overrides."""
        metrics_file = tmp_path / "gen_params_abc123" / "metrics.json"
        metrics_file.parent.mkdir(parents=True)
        metrics_file.write_text(json.dumps(metrics_data_simple))

        # Without optional fields - uses defaults from experiment_config
        results = parse_metrics_file(metrics_file, experiment_config)
        assert results[0]["model_key"] == "Llama-3.1-8B-Instruct"
        assert results[0]["source_uri"] == "meta-llama/Llama-3.1-8B-Instruct"

        # With optional fields - overrides defaults
        results = parse_metrics_file(
            metrics_file,
            experiment_config,
            model_key="CustomModel",
            source_uri="gs://custom/path",
        )
        assert results[0]["model_key"] == "CustomModel"
        assert results[0]["source_uri"] == "gs://custom/path"
