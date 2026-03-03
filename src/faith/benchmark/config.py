# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions to load benchmark configurations from paths or names."""

from pathlib import Path

from faith._internal.io.yaml import read_extended_yaml_file
from faith._internal.types.configs import Configuration


def load_config_from_path(benchmark_path: Path) -> Configuration:
    """Load a benchmark configuration from a benchmark path."""
    assert (
        benchmark_path.exists() and benchmark_path.is_dir()
    ), f"Benchmark path '{benchmark_path}' is not an existing directory."
    config_path = benchmark_path / "benchmark.yaml"
    assert (
        config_path.exists() and config_path.is_file()
    ), f"Benchmark config file '{config_path}' does not exist."
    return read_extended_yaml_file(config_path)["benchmark"]
