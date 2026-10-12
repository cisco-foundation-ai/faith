# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Deployment overrides cannot weaken a benchmark's workspace policy."""

from pathlib import Path
from unittest.mock import patch

import pytest

from faith._internal.io.resources import benchmarks_root
from faith._types.config.agentic import AgenticEnvironmentConfig
from faith.agentic.environments import DockerEnvironment
from faith.agentic.orchestration import build_environment
from faith.agentic.registries import _ConfiguredRegistry
from faith.benchmark.config import load_config_from_path


class _ShellOnly:
    """Serves command tools but cannot establish filesystem policy."""

    def run_command(self, command: str) -> tuple[int, str]:
        """Echo the command without executing it."""
        return 0, command


class _PolicyTestEnvironments(_ConfiguredRegistry):
    DOCKER = (DockerEnvironment,)
    SHELL_ONLY = (_ShellOnly,)


@pytest.mark.parametrize("phase", ["a", "b"])
@pytest.mark.parametrize(
    "deployment",
    [
        None,
        AgenticEnvironmentConfig(
            "docker",
            {
                "agent_image": "registry.example/vulnloc:v1",
                "read_only_workspace": False,
            },
        ),
        AgenticEnvironmentConfig(
            "kubernetes", {"warmpool": "test-pool", "read_only_workspace": False}
        ),
    ],
    ids=["benchmark-default", "docker-override", "kubernetes-override"],
)
def test_vulnloc_policy_survives_backend_selection(phase, deployment) -> None:
    config = load_config_from_path(
        benchmarks_root() / f"vulnerability-localization-{phase}", deployment
    ).agentic_config
    assert config is not None and config.read_only_workspace is True
    expected_backend = deployment.type if deployment is not None else "docker"
    assert config.environment.type == expected_backend
    with patch("faith.agentic.orchestration.AgenticEnvironmentType") as registry:
        build_environment(config)
    registry.assert_called_once_with(expected_backend)
    injected = registry.return_value.create.call_args.args[0]
    assert injected["read_only_workspace"] is True
    assert injected["working_subdir"] == "repo"


@pytest.mark.parametrize("phase", ["a", "b"])
def test_shell_support_alone_cannot_satisfy_a_read_only_benchmark(phase) -> None:
    with patch(
        "faith.agentic.registries.AgenticEnvironmentType", _PolicyTestEnvironments
    ):
        with pytest.raises(ValueError, match="does not support read_only_workspace"):
            load_config_from_path(
                benchmarks_root() / f"vulnerability-localization-{phase}",
                AgenticEnvironmentConfig("shell_only"),
            )


@pytest.mark.parametrize("backend", ["docker", "kubernetes"])
def test_backend_selection_keeps_writable_benchmarks_writable(backend) -> None:
    benchmark = Path(__file__).parents[1] / "experiment/testdata/bench-agentic"
    config = load_config_from_path(
        benchmark, AgenticEnvironmentConfig(backend)
    ).agentic_config
    assert config is not None and config.read_only_workspace is False
    with patch("faith.agentic.orchestration.AgenticEnvironmentType") as registry:
        build_environment(config)
    assert (
        registry.return_value.create.call_args.args[0]["read_only_workspace"] is False
    )
