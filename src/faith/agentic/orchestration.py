# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Wiring for agentic benchmarks: resolve config into runnable objects.

An agentic benchmark.yaml names its task loader, environment, tools, and answer
extractor by short name; these builders turn a config section into an instance by
resolving the name through `agentic.registries` (the runner assembles the results
into a ReActLoop).
"""

from faith._types.config.agentic import AgenticAnswerExtractorConfig, AgenticConfig
from faith.agentic.core import (
    AnswerExtractor,
    Environment,
    TaskLoader,
    ToolRegistry,
)
from faith.agentic.registries import (
    AgenticAnswerExtractorType,
    AgenticEnvironmentType,
    AgenticTaskLoaderType,
    AgenticToolType,
)

_SANDBOX_TTL_GRACE_SECONDS = 600


def build_task_loader(config: AgenticConfig) -> TaskLoader:
    """Resolve and construct the task loader, told where the agent will work.

    Takes the whole agentic config rather than its `task_loader` section, because
    `working_dir` is a sibling of that section: it is what the loader's prompt and
    the environment have to agree about, so neither owns it.
    """
    loader = config.task_loader
    return AgenticTaskLoaderType(loader.type).create(loader.config, config.working_dir)


def build_environment(config: AgenticConfig) -> Environment:
    """Resolve and construct the environment, told the settings it does not own.

    These are injected into the environment's config rather than passed as
    arguments: `Environment` declares only a lifecycle, and adding a constructor to
    it to carry them would make every environment's `__init__` part of the ABC.
    They are also declared outside the `environment` section, which a deployment
    may replace wholesale — so the injection is what makes them survive a backend
    switch, not merely a convenience.

    The derived TTL goes in first so an authored one wins: a cluster's cap on how
    long a claim may live is not the benchmark's to widen.
    """
    env = config.environment
    return AgenticEnvironmentType(env.type).create(
        {
            "shutdown_after_seconds": _sandbox_ttl(config.limits.timeout_seconds),
            **env.config,
            "working_subdir": config.working_subdir,
            "command_timeout_seconds": config.limits.command_timeout_seconds,
            "read_only_workspace": config.read_only_workspace,
        }
    )


def _sandbox_ttl(timeout_seconds: float | None) -> int | None:
    """A claim's self-destruct deadline, derived from the task's wall-clock budget.

    None without a budget: the deadline is absolute, so a guessed one would delete
    a sandbox still being worked in, which recovery papers over by rebuilding it —
    and a rebuild resets the workspace.

    The grace covers what the budget does not. `timeout_seconds` is checked between
    turns, so the turn that overruns it runs to completion, then teardown follows.
    """
    if timeout_seconds is None:
        return None
    return int(timeout_seconds) + _SANDBOX_TTL_GRACE_SECONDS


def build_answer_extractor(config: AgenticAnswerExtractorConfig) -> AnswerExtractor:
    """Resolve and construct the answer extractor named by an agentic extractor config."""
    return AgenticAnswerExtractorType(config.type).create(config.config)


def build_tool_registry(tool_names: list[str]) -> ToolRegistry:
    """Build a ToolRegistry holding the tools named in an agentic config.

    Each name resolves through `AgenticToolType`; duplicate names raise via the
    registry's own uniqueness check.
    """
    return ToolRegistry(AgenticToolType(name).create() for name in tool_names)
