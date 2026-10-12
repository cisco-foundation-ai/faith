# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for loading a benchmark config and a deployment's environment override.

The merge rules themselves belong to `resolve_environment` and are covered by
tests/_types/config/test_agentic.py; what is tested here is the loading seam —
which benchmarks an override reaches, and what a bad override file reports.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from faith._types.config.agentic import AgenticConfig, AgenticEnvironmentConfig
from faith.agentic.environments import DockerEnvironment
from faith.agentic.registries import _ConfiguredRegistry
from faith.benchmark.config import load_config_from_path, load_environment_config

_TESTDATA = Path(__file__).parent / "testdata" / "env-config"

# Left unexpanded by the loader: `resolve_resource_path` expands it when the
# environment is constructed, which is also where a bad path is reported.
_VULN_LOC_DOCKERFILE = "$BENCHMARKS_ROOT/vulnerability-localization/Dockerfile"


class _FakeEnvType(_ConfiguredRegistry):
    """Docker plus an environment satisfying no capability at all.

    Docker is kept so the benchmark's own section still resolves; the merge is
    what the second member is there to be rejected by.
    """

    DOCKER = (DockerEnvironment,)
    NO_CAPABILITIES = (object,)


def _benchmark_dir(name: str, tmp_path: Path) -> Path:
    """Install a testdata yaml into `tmp_path` as the benchmark.yaml of a directory."""
    shutil.copy(_TESTDATA / f"{name}.yaml", tmp_path / "benchmark.yaml")
    return tmp_path


def _environment(name: str) -> AgenticEnvironmentConfig:
    return load_environment_config(_TESTDATA / f"{name}.yaml")


def test_a_benchmark_without_an_override_keeps_its_own_environment(
    tmp_path: Path,
) -> None:
    """The common case: `faith query` with no extra file runs what the yaml declares."""
    config = load_config_from_path(_benchmark_dir("agentic-docker", tmp_path))

    assert config.agentic_config is not None
    assert config.agentic_config.environment == AgenticEnvironmentConfig(
        type="docker",
        config={"agent_dockerfile": _VULN_LOC_DOCKERFILE, "memory_limit": "4g"},
    )


@pytest.mark.parametrize(
    ("deployment", "expected"),
    [
        # Both keys survive the merge: the deployment's image is what gets pulled,
        # and the environment reads the pair that way rather than the merge picking.
        (
            "deployment-docker",
            AgenticEnvironmentConfig(
                type="docker",
                config={
                    "agent_dockerfile": _VULN_LOC_DOCKERFILE,
                    "agent_image": "registry.example/vulnerability-localization:v3",
                    "memory_limit": "4g",
                },
            ),
        ),
        (
            "deployment-kubernetes",
            AgenticEnvironmentConfig(
                type="kubernetes",
                config={
                    "warmpool": "vulnerability-localization-pool",
                    "namespace": "sandbox-pool",
                },
            ),
        ),
        # The loader accepts `type` case-insensitively, so the merge has to read it
        # the same way: a mixed-case spelling keeps the benchmark's `agent_dockerfile`
        # rather than reading as a switch to a backend that cannot use it.
        (
            "deployment-mixed-case",
            AgenticEnvironmentConfig(
                type="docker",
                config={
                    "agent_dockerfile": _VULN_LOC_DOCKERFILE,
                    "memory_limit": "16g",
                },
            ),
        ),
    ],
    ids=["same_type", "type_switch", "mixed_case_same_type"],
)
def test_a_deployment_override_reaches_an_agentic_benchmarks_environment(
    deployment: str, expected: AgenticEnvironmentConfig, tmp_path: Path
) -> None:
    config = load_config_from_path(
        _benchmark_dir("agentic-docker", tmp_path), _environment(deployment)
    )

    assert config.agentic_config is not None
    assert config.agentic_config.environment == expected


def test_a_deployment_override_leaves_the_rest_of_the_agentic_section_alone(
    tmp_path: Path,
) -> None:
    """Only the environment moves: a run on another backend is the same benchmark."""
    benchmark_dir = _benchmark_dir("agentic-docker", tmp_path)
    original = load_config_from_path(benchmark_dir).agentic_config
    merged = load_config_from_path(
        benchmark_dir, _environment("deployment-kubernetes")
    ).agentic_config

    assert original is not None and merged is not None
    assert merged == AgenticConfig(
        task_loader=original.task_loader,
        environment=merged.environment,
        answer_extractor=original.answer_extractor,
        tools=original.tools,
        limits=original.limits,
    )


def test_a_benchmark_that_both_builds_and_pulls_is_rejected_naming_its_path(
    tmp_path: Path,
) -> None:
    """The pair is illegal in an authored file, and this is the first of the two.

    Reported against the benchmark.yaml rather than the merged section, so the
    author is pointed at the file they wrote.
    """
    benchmark_dir = _benchmark_dir("agentic-builds-and-pulls", tmp_path)

    with pytest.raises(ValueError) as excinfo:
        load_config_from_path(benchmark_dir)

    assert str(excinfo.value).startswith(
        f"Benchmark config '{benchmark_dir / 'benchmark.yaml'}' sets both "
        f"'agent_dockerfile' and 'agent_image'."
    )


def test_a_deployment_that_both_builds_and_pulls_is_rejected_naming_its_path() -> None:
    """The other authored file. Checked as it is read, before it reaches a merge."""
    config_path = _TESTDATA / "deployment-builds-and-pulls.yaml"

    with pytest.raises(ValueError) as excinfo:
        load_environment_config(config_path)

    assert str(excinfo.value).startswith(
        f"Environment config '{config_path}' sets both 'agent_dockerfile' and "
        f"'agent_image'."
    )


def test_an_override_is_ignored_by_a_non_agentic_benchmark(tmp_path: Path) -> None:
    """A QA benchmark has no environment, so loading it must not fail on one."""
    config = load_config_from_path(
        _benchmark_dir("qa", tmp_path), _environment("deployment-kubernetes")
    )

    assert config.agentic_config is None
    assert config.metadata.name == "env-config-qa"


@patch("faith.agentic.registries.AgenticEnvironmentType", _FakeEnvType)
def test_a_merged_environment_is_revalidated_against_the_benchmarks_tools(
    tmp_path: Path,
) -> None:
    """A deployment can switch to a backend the benchmark's tools cannot use.

    Caught by construction: rebuilding the frozen agentic section re-runs
    `validate_tool_capabilities`, still before any container or cluster is touched.
    Both real environments serve every tool, so the mismatch needs a fake registry.
    """
    with pytest.raises(ValueError, match="does not support tool 'bash_exec'"):
        load_config_from_path(
            _benchmark_dir("agentic-docker", tmp_path),
            AgenticEnvironmentConfig(type="no_capabilities"),
        )


def test_an_environment_config_file_is_read_as_a_type_and_its_config() -> None:
    assert _environment("deployment-kubernetes") == AgenticEnvironmentConfig(
        type="kubernetes",
        config={
            "warmpool": "vulnerability-localization-pool",
            "namespace": "sandbox-pool",
        },
    )


def test_a_missing_environment_config_file_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_environment_config(missing)
    assert str(excinfo.value) == f"Environment config '{missing}' does not exist."


@pytest.mark.parametrize(
    ("deployment", "expected"),
    [
        ("deployment-untyped", " must be a YAML mapping naming an environment 'type'."),
        (
            "deployment-not-a-mapping",
            " must be a YAML mapping naming an environment 'type'.",
        ),
        (
            "deployment-unregistered",
            ": 'kubernettes' is not a valid AgenticEnvironmentType",
        ),
        (
            "deployment-scalar-config",
            " has a 'config' that is not a YAML mapping of environment settings.",
        ),
    ],
    ids=["no_type", "not_a_mapping", "unregistered_type", "scalar_config"],
)
def test_a_malformed_environment_config_file_is_rejected_naming_the_path(
    deployment: str, expected: str
) -> None:
    """Each error names the file, since the deployment wrote it, not the benchmark."""
    config_path = _TESTDATA / f"{deployment}.yaml"
    with pytest.raises(ValueError) as excinfo:
        load_environment_config(config_path)
    assert str(excinfo.value) == f"Environment config '{config_path}'{expected}"


def test_a_deployment_with_every_setting_commented_out_overrides_nothing() -> None:
    """`config:` decodes to None, not to the field default; it must read as empty.

    The natural mid-edit state of a deployment file, and the one that reached the
    merge as a null and failed there with neither the file nor the key named.
    """
    assert _environment("deployment-bare-config") == AgenticEnvironmentConfig(
        type="docker", config={}
    )
