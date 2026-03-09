# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for FAITH metrics parser."""

import json
from pathlib import Path
from unittest.mock import ANY

import pytest

from faith.ingestion.parser import (
    ExperimentConfig,
    parse_experiment_config,
    parse_metrics_file,
)

_TESTDATA_DIR = Path(__file__).parent / "testdata"

_EXPERIMENT_DATA = {
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


def test_parse_config():
    """Test parsing experiment config."""
    config, primary_metric = parse_experiment_config(_EXPERIMENT_DATA)

    assert config == ExperimentConfig(
        model_key="Llama-3.1-8B-Instruct",  # gitleaks:allow
        source_uri="meta-llama/Llama-3.1-8B-Instruct",
        benchmark="ctibench-mcqa",
        temperature=0.3,
        top_p=1.0,
        max_completion_tokens=1024,
        context_length=3500,
        generation_mode="chat_comp",
        prompt_format="chat",
        num_shots=0,
        num_shots_pool_size=1,
        num_trials=10,
        seed=42,
        sample_size=100,
    )
    assert primary_metric == "accuracy.mean"


def test_parse_missing_benchmark():
    """Test parsing config with incomplete benchmark spec raises error."""
    data = {"experiment_params": {"benchmark": {}, "model": {"path": "test/model"}}}

    # BenchmarkSpec.from_dict() will raise KeyError for missing required fields
    with pytest.raises(KeyError):
        parse_experiment_config(data)


def test_parse_num_shots_fractional():
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

    config, _ = parse_experiment_config(data)
    assert config == ExperimentConfig(
        model_key="test/model",
        source_uri="test/model",
        benchmark="test-bench",
        temperature=None,
        top_p=None,
        max_completion_tokens=None,
        context_length=None,
        generation_mode="chat_comp",
        prompt_format="chat",
        num_shots=3,
        num_shots_pool_size=5,
        num_trials=None,
        seed=None,
        sample_size=None,
    )


def test_parse_primary_metric_from_config():
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

    _, primary_metric = parse_experiment_config(data)
    assert primary_metric == "custom_metric.mean"


def test_parse_missing_primary_metric():
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

    _, primary_metric = parse_experiment_config(data)
    assert primary_metric is None


_EXPERIMENT_CONFIG = ExperimentConfig(
    model_key="Llama-3.1-8B-Instruct",  # gitleaks:allow
    source_uri="meta-llama/Llama-3.1-8B-Instruct",
    benchmark="ctibench-mcqa",
    temperature=0.3,
    top_p=1.0,
    max_completion_tokens=1024,
    context_length=3500,
    generation_mode="chat_comp",
    prompt_format="chat",
    num_shots=0,
    num_shots_pool_size=1,
    num_trials=10,
    seed=42,
    sample_size=100,
)
_PRIMARY_METRIC_NAME = "accuracy.mean"


def test_parse_simple_metrics():
    """Test parsing simple flat metrics."""
    metrics_file = _TESTDATA_DIR / "simple" / "metrics.json"

    results = parse_metrics_file(metrics_file, _EXPERIMENT_CONFIG, _PRIMARY_METRIC_NAME)

    expected = [
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "accuracy.mean",
            "metric_value": 0.8875,
            "is_primary": True,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "accuracy.std",
            "metric_value": 0.0096,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "accuracy.min",
            "metric_value": 0.8625,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "accuracy.max",
            "metric_value": 0.9,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "weighted_avg_f1.mean",
            "metric_value": 0.8866,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "weighted_avg_f1.std",
            "metric_value": 0.01,
            "is_primary": False,
        },
    ]

    assert results == expected


def test_parse_nested_metrics():
    """Test parsing nested metrics (like f1_scores)."""
    metrics_file = _TESTDATA_DIR / "nested" / "metrics.json"

    results = parse_metrics_file(metrics_file, _EXPERIMENT_CONFIG, _PRIMARY_METRIC_NAME)

    expected = [
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "accuracy.mean",
            "metric_value": 0.4608,
            "is_primary": True,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "accuracy.std",
            "metric_value": 0.0,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "f1_scores.A.mean",
            "metric_value": 0.511,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "f1_scores.A.std",
            "metric_value": 0.0,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "f1_scores.B.mean",
            "metric_value": 0.650,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "f1_scores.B.std",
            "metric_value": 0.0,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "f1_scores.C.mean",
            "metric_value": 0.659,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "f1_scores.C.std",
            "metric_value": 0.0,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "f1_scores.D.mean",
            "metric_value": 0.511,
            "is_primary": False,
        },
        {
            "metrics_file_uri": ANY,
            "faith_version": ANY,
            "ingest_time": ANY,
            "model_key": "Llama-3.1-8B-Instruct",  # gitleaks:allow
            "source_uri": "meta-llama/Llama-3.1-8B-Instruct",
            "benchmark": "ctibench-mcqa",
            "temperature": 0.3,
            "top_p": 1.0,
            "max_completion_tokens": 1024,
            "context_length": 3500,
            "generation_mode": "chat_comp",
            "prompt_format": "chat",
            "num_trials": 10,
            "num_shots": 0,
            "num_shots_pool_size": 1,
            "seed": 42,
            "sample_size": 100,
            "metric_name": "f1_scores.D.std",
            "metric_value": 0.0,
            "is_primary": False,
        },
    ]

    assert results == expected


def test_parse_invalid_stats_type(tmp_path):
    """Test parsing metrics.json with invalid stats type returns empty list."""
    metrics_file = tmp_path / "gen_params_abc123" / "metrics.json"
    metrics_file.parent.mkdir(parents=True)
    metrics_file.write_text(json.dumps({"stats": "not a dict"}))

    results = parse_metrics_file(metrics_file, _EXPERIMENT_CONFIG, _PRIMARY_METRIC_NAME)
    assert results == []


def test_parse_nonexistent_file(tmp_path):
    """Test parsing nonexistent metrics file."""
    with pytest.raises(FileNotFoundError):
        parse_metrics_file(
            tmp_path / "nonexistent.json", _EXPERIMENT_CONFIG, _PRIMARY_METRIC_NAME
        )
