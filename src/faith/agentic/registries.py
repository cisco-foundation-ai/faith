# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Short-name registries for the pluggable pieces of an agentic benchmark.

An agentic benchmark.yaml names its task loader, environment, tools, and answer
extractor by short name; these closed CIEnum registries map those names to
implementing classes, mirroring `DomainSpecificScore`. `agentic.orchestration`
turns a resolved class into an instance.

Kept separate from that module, and importing nothing from
`faith._types.config`, so the config layer can validate a benchmark.yaml against
these registries at parse time without a cycle.
"""

from typing import Any

from faith._types.enums import CIEnum
from faith.agentic.core import Tool
from faith.agentic.core.capabilities import ReadOnlyWorkspace
from faith.agentic.environments import DockerEnvironment, KubernetesEnvironment
from faith.agentic.extractors import VulnerabilityLocalizationAnswerExtractor
from faith.agentic.loaders import VulnerabilityLocalizationTaskLoader
from faith.agentic.tools import (
    BashExecTool,
    SubmitNoVulnerabilityFoundTool,
    SubmitTool,
    SubmitVulnerableFilesTool,
    TerminalTool,
)


class _ConfiguredRegistry(CIEnum):
    """Base for short-name registries whose classes take a config dict.

    Each member's value wraps the implementing class in a 1-tuple, as
    `DomainSpecificScore` does, so resolution is a name lookup plus a one-argument
    construction.
    """

    @property
    def impl_cls(self) -> type:
        """The class this member registers."""
        return self.value[0]

    def create(self, config: dict) -> Any:
        """Instantiate the registered class from its benchmark.yaml config."""
        return self.impl_cls(config)


class _WorkingDirRegistry(CIEnum):
    """Base for registries whose classes take a config dict and the working dir.

    A separate base rather than an argument on `_ConfiguredRegistry.create`: what a
    registry base states is which arguments its implementations' `__init__` takes.
    """

    @property
    def impl_cls(self) -> type:
        """The class this member registers."""
        return self.value[0]

    def create(self, config: dict, working_dir: str) -> Any:
        """Instantiate the registered class from its config and working directory."""
        return self.impl_cls(config, working_dir)


class _NoConfigRegistry(CIEnum):
    """Base for registries whose classes are constructed with no arguments."""

    @property
    def impl_cls(self) -> type[Tool]:
        """The tool class this member registers."""
        return self.value[0]

    def create(self) -> Tool:
        """Instantiate the registered tool."""
        return self.impl_cls()


class AgenticTaskLoaderType(_WorkingDirRegistry):
    """Registry of TaskLoader implementations, keyed by short name.

    Loaders take the agent's working directory alongside their config, since a
    prompt that names the directory must name the one the agent runs in.
    """

    # Reads the benchmark's manifest CSV, fetching each row's source from GitHub.
    VULNERABILITY_LOCALIZATION = (VulnerabilityLocalizationTaskLoader,)


class AgenticEnvironmentType(_ConfiguredRegistry):
    """Registry of Environment implementations, keyed by short name.

    Each is constructed as ``EnvClass(config)`` from the environment section's
    ``config`` sub-dict (image, limits, timeouts).
    """

    DOCKER = (DockerEnvironment,)  # Run the agent in an isolated Docker container.
    KUBERNETES = (KubernetesEnvironment,)  # Run the agent in a k8s Agent Sandbox.


class AgenticAnswerExtractorType(_ConfiguredRegistry):
    """Registry of AnswerExtractor implementations, keyed by short name."""

    # Reads the benchmark's ranked file list, and an abstention as an empty list.
    VULNERABILITY_LOCALIZATION = (VulnerabilityLocalizationAnswerExtractor,)


class AgenticToolType(_NoConfigRegistry):
    """Registry of Tool implementations, keyed by the tool's function-call name.

    Each member name matches the wrapped tool's `name`, so a benchmark.yaml naming
    "bash_exec" resolves to BASH_EXEC. The first two members are a starter set: a
    benchmark registers its own and may use none of these. The loop's only
    requirement is that some registered tool is terminal (`Tool.is_terminal`).
    """

    BASH_EXEC = (BashExecTool,)  # Run an unrestricted shell command.
    SUBMIT = (SubmitTool,)  # Submit a string answer, terminating the run.
    # The benchmark's three tools: bash_exec under the name its system prompt uses,
    # then name the vulnerable files, or abstain.
    TERMINAL = (TerminalTool,)
    SUBMIT_VULNERABLE_FILES = (SubmitVulnerableFilesTool,)
    SUBMIT_NO_VULNERABILITY_FOUND = (SubmitNoVulnerabilityFoundTool,)


def validate_read_only_workspace(environment_type: str, required: bool) -> None:
    """Reject a required policy the environment class cannot implement."""
    if required and not issubclass(
        AgenticEnvironmentType(environment_type).impl_cls, ReadOnlyWorkspace
    ):
        raise ValueError(
            f"Environment '{environment_type}' does not support read_only_workspace"
        )


def validate_tool_capabilities(environment_type: str, tool_names: list[str]) -> None:
    """Reject a tool the named environment cannot serve.

    Resolves both names through the registries and compares each tool's
    ``required_capabilities`` against the environment CLASS, so no container or
    cluster is touched. Runs at config parse time, before an eval starts.

    Args:
        environment_type: The environment section's ``type`` short name.
        tool_names: The tool short names the benchmark enables.

    Raises:
        ValueError: If a name is unregistered, or a tool needs a capability the
            environment class does not implement.
    """
    env_cls = AgenticEnvironmentType(environment_type).impl_cls
    for name in tool_names:
        tool_cls = AgenticToolType(name).impl_cls
        if missing := tuple(
            cap
            for cap in tool_cls.required_capabilities
            if not issubclass(env_cls, cap)
        ):
            required = ", ".join(cap.__name__ for cap in missing)
            raise ValueError(
                f"Environment '{environment_type}' does not support tool '{name}' "
                f"(requires {required}). Configured tools: {', '.join(tool_names)}."
            )
