# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration types for agentic benchmarks.

Agentic benchmarks run an autonomous agent inside a sandboxed environment,
letting it call tools (bash, file reads, etc.) to accomplish a task, then score
the agent's submitted answer. These types capture the sections of a
benchmark.yaml that configure that flow.
"""

import posixpath
from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import DataClassJsonMixin
from dataclasses_json import config as dc_config

from faith.agentic.core import DEFAULT_WORKSPACE_DIR
from faith.agentic.core.react_loop import validate_agent_limits


@dataclass(frozen=True)
class AgenticTaskLoaderConfig(DataClassJsonMixin):
    """Configuration for loading agentic benchmark tasks.

    Attributes:
        type: Registered short name of the TaskLoader, resolved through
            AgenticTaskLoaderType — not an import path.
        config: Loader-specific configuration passed to the loader's __init__.
    """

    type: str
    config: dict[str, Any] = field(
        default_factory=dict, metadata=dc_config(exclude=lambda x: not x)
    )


@dataclass(frozen=True)
class AgenticEnvironmentConfig(DataClassJsonMixin):
    """Configuration for the agent's execution environment.

    Attributes:
        type: Registered short name of the Environment, resolved through
            AgenticEnvironmentType — not an import path.
        config: Environment-specific configuration (image, limits, mount path).
    """

    type: str
    config: dict[str, Any] = field(
        default_factory=dict, metadata=dc_config(exclude=lambda x: not x)
    )


@dataclass(frozen=True)
class AgenticAnswerExtractorConfig(DataClassJsonMixin):
    """Configuration for the agentic answer extractor.

    The AnswerExtractor named here derives a prediction from the agent's
    trajectory; the metric math reuses QA's `output_processing.score_fns` /
    `primary_metric`, not this section.

    Attributes:
        type: Registered short name of the AnswerExtractor, resolved through
            AgenticAnswerExtractorType — not an import path.
        config: Extractor-specific configuration passed to the extractor's
            __init__. NOT metric config — metrics live in output_processing.
    """

    type: str
    config: dict[str, Any] = field(
        default_factory=dict, metadata=dc_config(exclude=lambda x: not x)
    )


@dataclass(frozen=True)
class AgenticLimitsConfig(DataClassJsonMixin):
    """Configuration for agent evaluation limits.

    Attributes:
        max_tool_calls: Maximum number of tool calls the agent may make per task.
        max_turns: Hard cap on model turns, the safety-net for unproductive turns
            (e.g. turns that only emit malformed calls and never spend the tool
            budget). Must be >= max_tool_calls.
        max_consecutive_no_tool_calls: How many prose-only turns in a row end the
            run as NO_TOOL_CALLS. Earlier ones are nudged instead, so one chatty
            turn does not discard the task. 1 restores fail-on-first behavior.
        timeout_seconds: Supplementary per-task wall-clock budget in seconds, or
            None for unbounded. max_tool_calls and max_turns already guarantee
            termination; this only catches slow/stuck runs, and a sensible value
            is deployment-specific, so it defaults to None (set per benchmark).
        command_timeout_seconds: Bound on each individual command the agent runs,
            in seconds; 0 for no bound. The inner bound to timeout_seconds' outer
            one, declared here rather than under `environment` because every
            backend enforces it — a section a deployment may replace wholesale is
            no place for a setting that is not backend-specific.
    """

    max_tool_calls: int = 50
    max_turns: int = 100
    max_consecutive_no_tool_calls: int = 3
    timeout_seconds: float | None = field(
        default=None, metadata=dc_config(exclude=lambda x: x is None)
    )
    command_timeout_seconds: int = 60

    def __post_init__(self) -> None:
        validate_agent_limits(
            self.max_tool_calls,
            self.max_turns,
            self.max_consecutive_no_tool_calls,
            self.timeout_seconds,
        )
        if self.command_timeout_seconds < 0:
            raise ValueError(
                f"command_timeout_seconds ({self.command_timeout_seconds}) must be "
                f"non-negative; use 0 for no bound."
            )


@dataclass(frozen=True)
class AgenticConfig(DataClassJsonMixin):
    """Top-level configuration for an agentic benchmark.

    Groups the sections of a benchmark.yaml that only apply to agentic
    benchmarks (those carrying an `agentic_config`): how tasks are loaded, where
    the agent runs, which tools it may call, and its evaluation limits.

    Each section that selects an implementation is named for the abstraction it
    selects and picks one with `type`.

    Attributes:
        task_loader: How agentic tasks are loaded (loader type and its config).
        environment: Where the agent runs (environment type and its config).
        answer_extractor: How a prediction is derived from the agent's trajectory.
        working_subdir: Subdirectory of the workspace the agent works from, for a
            benchmark whose data unpacks into one; "" to work from the workspace
            root. Declared here rather than under `environment` because the task
            loader names the same directory in its prompt and the environment
            runs the agent's commands in it, and the two must agree.
        tools: Names of the tools the agent may call.
        system_prompt: Standing instructions seeded ahead of the task prompt —
            the benchmark's role framing, method, and effort expectations. A plain
            string rather than a template: agentic tasks have no render variable,
            and per-task text belongs in the loader, where the row data is. None
            or empty sends no system message.
        limits: Evaluation limits (tool-call, turn, prose-turn, and time budgets).
    """

    task_loader: AgenticTaskLoaderConfig
    environment: AgenticEnvironmentConfig
    answer_extractor: AgenticAnswerExtractorConfig
    working_subdir: str = field(default="", metadata=dc_config(exclude=lambda x: not x))
    tools: list[str] = field(
        default_factory=list, metadata=dc_config(exclude=lambda x: not x)
    )
    system_prompt: str | None = field(
        default=None, metadata=dc_config(exclude=lambda x: x is None)
    )
    limits: AgenticLimitsConfig = field(default_factory=AgenticLimitsConfig)

    def __post_init__(self) -> None:
        # A bare `working_subdir:` key in yaml decodes to None, not to the field
        # default, so neither the join below nor an environment has to defend
        # against it.
        if self.working_subdir is None:
            object.__setattr__(self, "working_subdir", "")
        # Checked on the derived path, so an absolute subdir and one that climbs
        # out with `..` are rejected by the same test. The workspace root itself
        # is the no-subdir case, not an escape.
        if (
            self.working_dir != DEFAULT_WORKSPACE_DIR
            and not self.working_dir.startswith(f"{DEFAULT_WORKSPACE_DIR}/")
        ):
            raise ValueError(
                f"working_subdir {self.working_subdir!r} does not name a "
                f"directory inside the workspace (resolved to "
                f"{self.working_dir!r}); it must be relative to "
                f"{DEFAULT_WORKSPACE_DIR}."
            )

    @property
    def working_dir(self) -> str:
        """The absolute sandbox directory the agent's commands run in.

        Derived rather than authored, so a prompt naming this directory and the
        environment running the agent there cannot disagree.
        """
        return posixpath.normpath(
            posixpath.join(DEFAULT_WORKSPACE_DIR, self.working_subdir)
        )
