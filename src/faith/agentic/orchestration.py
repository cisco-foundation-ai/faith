# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Wiring for agentic benchmarks: resolve config into runnable objects.

An agentic benchmark.yaml names its task loader, environment, tools, and answer
extractor by short name; this module turns those names into instances, which the
trial runner assembles into a ReActLoop. Selection uses closed CIEnum registries
keyed by short name, mirroring `DomainSpecificScore`.
"""

from typing import Any

from faith._types.config.agentic import AgenticAnswerExtractorConfig, AgenticConfig
from faith._types.enums import CIEnum
from faith.agentic.core import (
    AnswerExtractor,
    Environment,
    TaskLoader,
    Tool,
    ToolRegistry,
)
from faith.agentic.tools import BashExecTool, SubmitTool


class _ConfiguredRegistry(CIEnum):
    """Base for short-name registries of config-constructed classes.

    Each member's value wraps the implementing class in a 1-tuple, as
    `DomainSpecificScore` does, and every implementation takes its benchmark.yaml
    `config` sub-dict as a single opaque argument, so resolution is a name lookup
    plus a one-argument construction.
    """

    @property
    def _impl_cls(self) -> type:
        return self.value[0]

    def create(self, config: dict) -> Any:
        """Instantiate the registered class from its benchmark.yaml config."""
        return self._impl_cls(config)


class _WorkingDirRegistry(CIEnum):
    """Base for short-name registries of classes also told where the agent works.

    A separate base rather than an argument on `_ConfiguredRegistry.create`,
    following `_NoConfigRegistry`: which arguments an implementation's `__init__`
    takes is what a registry base states.
    """

    @property
    def _impl_cls(self) -> type:
        return self.value[0]

    def create(self, config: dict, working_dir: str) -> Any:
        """Instantiate the registered class from its config and working directory."""
        return self._impl_cls(config, working_dir)


class _NoConfigRegistry(CIEnum):
    """Base for short-name registries of classes constructed with no arguments."""

    @property
    def _impl_cls(self) -> type[Tool]:
        return self.value[0]

    def create(self) -> Tool:
        """Instantiate the registered tool."""
        return self._impl_cls()


class AgenticTaskLoaderType(_WorkingDirRegistry):
    """Registry of TaskLoader implementations.

    Loaders take the agent's working directory alongside their config, since a
    prompt that names the directory must name the one the agent runs in.
    """


class AgenticEnvironmentType(_ConfiguredRegistry):
    """Registry of Environment implementations."""


class AgenticAnswerExtractorType(_ConfiguredRegistry):
    """Registry of AnswerExtractor implementations."""


class AgenticToolType(_NoConfigRegistry):
    """Registry of Tool implementations, keyed by the tool's function-call name.

    Tools take no configuration (only the environment is stateful), so this
    registry constructs them with no arguments rather than inheriting
    `_ConfiguredRegistry`'s config-taking `create`. Each member name matches the
    wrapped tool's `name`, so a benchmark.yaml naming "bash_exec" resolves to
    BASH_EXEC.
    """

    BASH_EXEC = (BashExecTool,)  # Run a shell command in the environment.
    SUBMIT = (SubmitTool,)  # Submit the final answer, terminating the run.


def build_task_loader(config: AgenticConfig) -> TaskLoader:
    """Resolve and construct the task loader, told where the agent will work.

    Takes the whole agentic config rather than its `task_loader` section, because
    `working_dir` is a sibling of that section: it is what the loader's prompt and
    the environment have to agree about, so neither owns it.
    """
    loader = config.task_loader
    return AgenticTaskLoaderType(loader.type).create(loader.config, config.working_dir)


def build_environment(config: AgenticConfig) -> Environment:
    """Resolve and construct the environment, told the benchmark's working subdir.

    The subdir is injected into the environment's config rather than passed as an
    argument: `Environment` declares only a lifecycle, and adding a constructor to
    it to carry a path would make every environment's `__init__` part of the ABC.
    """
    env = config.environment
    return AgenticEnvironmentType(env.type).create(
        {**env.config, "working_subdir": config.working_subdir}
    )


def build_answer_extractor(config: AgenticAnswerExtractorConfig) -> AnswerExtractor:
    """Resolve and construct the answer extractor named by an agentic extractor config."""
    return AgenticAnswerExtractorType(config.type).create(config.config)


def build_tool_registry(tool_names: list[str]) -> ToolRegistry:
    """Build a ToolRegistry holding the tools named in an agentic config.

    Each name resolves through `AgenticToolType`; duplicate names raise via the
    registry's own uniqueness check.
    """
    return ToolRegistry(AgenticToolType(name).create() for name in tool_names)
