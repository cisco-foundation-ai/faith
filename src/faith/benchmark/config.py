# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions to load benchmark configurations from paths or names."""

import dataclasses
from pathlib import Path

from faith._internal.io.yaml import read_extended_yaml_file
from faith._types.config.agentic import (
    AgenticEnvironmentConfig,
    resolve_environment,
    validate_authored_environment,
)
from faith._types.config.benchmark import BenchmarkConfig
from faith.agentic.registries import AgenticEnvironmentType


def load_config_from_path(
    benchmark_path: Path,
    environment_config: AgenticEnvironmentConfig | None = None,
) -> BenchmarkConfig:
    """Load a benchmark configuration from a benchmark path.

    Args:
        benchmark_path: Directory holding the benchmark.yaml.
        environment_config: A deployment's environment override, merged over an
            agentic benchmark's own environment section and ignored by any other
            benchmark. Defaults to None, which listing always passes: which
            environment a run would use says nothing about what benchmarks exist.
    """
    assert (
        benchmark_path.exists() and benchmark_path.is_dir()
    ), f"Benchmark path '{benchmark_path}' is not an existing directory."
    config_path = benchmark_path / "benchmark.yaml"
    assert (
        config_path.exists() and config_path.is_file()
    ), f"Benchmark config file '{config_path}' does not exist."
    config = BenchmarkConfig.from_dict(
        read_extended_yaml_file(config_path)["benchmark"]
    )
    if config.agentic_config is not None:
        # Checked here rather than in AgenticConfig, which is also rebuilt from the
        # merged config below — where holding both keys is legitimate.
        validate_authored_environment(
            config.agentic_config.environment.config,
            f"Benchmark config '{config_path}'",
        )
    return resolve_deployment_environment(config, environment_config)


def resolve_deployment_environment(
    config: BenchmarkConfig,
    environment_config: AgenticEnvironmentConfig | None,
) -> BenchmarkConfig:
    """Merge a deployment's environment override over a benchmark's own section.

    Args:
        config: A benchmark's configuration, as its author wrote it.
        environment_config: The deployment's override, or None to keep the authored
            environment.
    """
    if environment_config is None or config.agentic_config is None:
        return config
    # Rebuilding the agentic section re-runs its __post_init__, so the merged
    # environment is re-checked against the benchmark's tools. That is the whole of
    # it: the environment class is not constructed here, so its required keys are
    # still first checked at build_environment time.
    return dataclasses.replace(
        config,
        agentic_config=dataclasses.replace(
            config.agentic_config,
            environment=resolve_environment(
                config.agentic_config.environment, environment_config
            ),
        ),
    )


def load_environment_config(config_path: Path) -> AgenticEnvironmentConfig:
    """Load a deployment's agentic environment override from a YAML file.

    Read with the extended loader, so a deployment composes its file with `!from`
    exactly as a benchmark.yaml does. The `type` is resolved here so a typo fails
    before any benchmark is loaded, and every error names the file the deployment
    wrote.

    Raises:
        FileNotFoundError: If `config_path` is not a file.
        ValueError: If the file is not a mapping naming a registered `type`.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Environment config '{config_path}' does not exist.")
    config_dict = read_extended_yaml_file(config_path)
    if not isinstance(config_dict, dict) or "type" not in config_dict:
        raise ValueError(
            f"Environment config '{config_path}' must be a YAML mapping naming an "
            f"environment 'type'."
        )
    # Checked before `from_dict`, which reports a non-mapping as an AttributeError
    # from its own internals, naming neither this file nor the key.
    if not isinstance(config_dict.get("config", {}) or {}, dict):
        raise ValueError(
            f"Environment config '{config_path}' has a 'config' that is not a YAML "
            f"mapping of environment settings."
        )
    config = AgenticEnvironmentConfig.from_dict(config_dict)
    try:
        AgenticEnvironmentType(config.type)
    except ValueError as e:
        raise ValueError(f"Environment config '{config_path}': {e}") from e
    validate_authored_environment(config.config, f"Environment config '{config_path}'")
    return config
