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
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import Any

from dataclasses_json import DataClassJsonMixin
from dataclasses_json import config as dc_config

from faith._internal.io.yaml import deep_merge_config
from faith._types.model.engine import REDACTED, is_secret_name
from faith.agentic import registries
from faith.agentic.core import DEFAULT_WORKSPACE_DIR
from faith.agentic.core.react_loop import validate_agent_limits
from faith.agentic.environments.kubernetes import ConnectionMode
from faith.agentic.registries import (
    validate_read_only_workspace,
    validate_tool_capabilities,
)


def _read_only_boolean(value: Any) -> bool:
    """Do not let config decoding turn the string 'false' into True."""
    if not isinstance(value, bool):
        raise TypeError("read_only_workspace must be a boolean")
    return value


def _redact_environment_secrets(value: Any) -> Any:
    """Copy config, masking credential-named keys and NAME=value assignments.

    Uses the model-config naming policy, not value inspection: a secret with an
    innocuous name is not detected. Runtime config is never mutated or decoded
    from this representation.
    """
    if isinstance(value, dict):
        return {
            key: REDACTED if is_secret_name(key) else _redact_environment_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_environment_secrets(item) for item in value]
    if isinstance(value, str):
        name, separator, _ = value.partition("=")
        if separator and is_secret_name(name):
            return f"{name}={REDACTED}"
    return value


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
        config: Environment-specific configuration (image, limits, timeouts).
    """

    type: str
    config: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        metadata=dc_config(
            exclude=lambda x: not x, encoder=_redact_environment_secrets
        ),
    )

    def __post_init__(self) -> None:
        # A bare `config:` key in yaml decodes to None, not to the field default,
        # as a bare `tools:` does in AgenticConfig. Normalized here so neither the
        # merge nor an environment's __init__ has to defend against it.
        if self.config is None:
            object.__setattr__(self, "config", {})


def validate_authored_environment(config: dict[str, Any], source: str) -> None:
    """Reject an authored environment config a run would only fail on much later.

    Both checks are here rather than in the environment's ``__init__`` because that
    runs per trial, inside the run loop: a sweep over several benchmarks bills the
    earlier ones in full before reaching the invalid one, while every other config
    error costs nothing.

    Checked per authored file, not on a merged config: a deployment naming an image
    over a benchmark that builds one is the supported way to say "pull this instead",
    and that merge legitimately holds both keys.

    Args:
        config: An environment section's `config` sub-dict as it was written.
        source: What to name in the error — the file or section the author wrote.

    Raises:
        ValueError: If both image keys are set, or the sandbox connection mode is
            not a known one.
    """
    # `agent_dockerfile` means "build it, and FAITH names what it builds";
    # `agent_image` means "pull the one someone else published". Together they ask
    # for a local build under a name the author chose, which is how a build ends up
    # overwriting a tag that is not theirs, and how a mutable tag ends up standing
    # in for the recipe that produced it. Neither is undoable once the image exists.
    if config.get("agent_dockerfile") and config.get("agent_image"):
        raise ValueError(
            f"{source} sets both 'agent_dockerfile' and 'agent_image'. Set only "
            f"'agent_dockerfile' to build the agent image (FAITH names what it "
            f"builds, from the build context's digest), or only 'agent_image' to "
            f"pull a published one."
        )
    # Checked against the names rather than by constructing the environment, which
    # would need the optional k8s extra installed just to read a config file.
    connection = config.get("connection")
    if connection is not None:
        try:
            ConnectionMode(connection)
        except ValueError as e:
            raise ValueError(
                f"{source} sets 'connection' to {connection!r}; must be one of "
                f"{', '.join(ConnectionMode)}."
            ) from e


def resolve_environment(
    benchmark: AgenticEnvironmentConfig,
    deployment: AgenticEnvironmentConfig | None,
) -> AgenticEnvironmentConfig:
    """Merge a deployment's environment selection over a benchmark's default.

    A benchmark's environment section is a portable default — the one that works
    on a laptop — so a deployment that cannot run it replaces it.

    Named fields merge, the deployment winning wherever it sets one. The `config`
    dict merges only when the `type` matches; a type switch replaces it, since its
    keys name things the new backend cannot read (a cluster has no use for a
    Dockerfile). Every named field is merged generically, so adding one needs no
    change here.

    The `config` merge recurses, so a deployment adding one container env var
    keeps the benchmark's rest: a shallow merge would drop the siblings it never
    mentioned and report nothing. The cost is that a nested dict can only be added
    to, not cleared — `null` already means "unset, use the default" elsewhere in
    these yamls, so spending it here as "delete" would collide.

    Types are compared as registry members, not as strings: the registry resolves
    them case-insensitively, so `Docker` and `docker` name one backend and must
    merge rather than read as a switch that discards the benchmark's config. The
    resolved type is stored canonically, so a run's record names its environment
    the same way however the deployment spelled it.

    Every key merges the same way, this one included: a deployment naming an
    `agent_image` over a benchmark that ships an `agent_dockerfile` leaves both in
    the merged config, and the environment reads that pair as "pull the deployment's
    image". Nothing key-specific happens here, because the two keys are mutually
    exclusive in an authored file — which is what makes the merged combination
    unambiguous rather than something this function has to arbitrate.

    Args:
        benchmark: The environment section the benchmark.yaml declared.
        deployment: A deployment's override, or None to use the benchmark's as-is.

    Returns:
        The environment section the run should use.
    """
    if deployment is None:
        return benchmark
    overrides = {
        f.name: value
        for f in fields(deployment)
        if f.name != "config" and (value := getattr(deployment, f.name)) is not None
    }
    if "type" in overrides:
        # Resolved through the module rather than an imported name so a test that
        # substitutes the registry is seen here too, as it is in `registries`.
        env_type = registries.AgenticEnvironmentType(overrides["type"])
        overrides["type"] = str(env_type)
        same_type = env_type is registries.AgenticEnvironmentType(benchmark.type)
    else:
        same_type = True
    if same_type:
        config = deep_merge_config(deepcopy(benchmark.config), deployment.config)
    else:
        config = deployment.config
    return replace(benchmark, **overrides, config=config)


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
        recover_on_environment_failure: Whether a sandbox that dies mid-task is
            rebuilt and the failed call retried, unseen and uncharged. Rebuilding
            re-runs setup, so the workspace reverts: a benchmark whose agent writes
            files must set this false, or the agent silently continues without its
            own work.
    """

    max_tool_calls: int = 50
    max_turns: int = 100
    max_consecutive_no_tool_calls: int = 3
    timeout_seconds: float | None = field(
        default=None, metadata=dc_config(exclude=lambda x: x is None)
    )
    command_timeout_seconds: int = 60
    recover_on_environment_failure: bool = True

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

    This is where the environment and the tools first meet, so an environment that
    cannot serve a configured tool is rejected here rather than left to whichever
    caller happens to check.

    Attributes:
        task_loader: How agentic tasks are loaded (loader type and its config).
        environment: Where the agent runs (environment type and its config).
        answer_extractor: How a prediction is derived from the agent's trajectory.
        read_only_workspace: Require an immutable input workspace. Unsupported
            environments fail setup; writable scratch outside it is still allowed.
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
    read_only_workspace: bool = field(
        default=False,
        metadata=dc_config(decoder=_read_only_boolean, exclude=lambda x: x is False),
    )

    def to_identity_dict(self) -> dict[str, Any]:
        """Unredacted config for hashing only; never persist or log this dict.

        Keep the existing identity shape and values, including authored secrets.
        Redacting the record must not invalidate old checkpoints or collapse
        distinct authored configs onto one cache path.
        """
        identity: dict[str, Any] = self.to_dict()
        if self.environment.config:
            identity["environment"]["config"] = deepcopy(self.environment.config)
        return identity

    def __post_init__(self) -> None:
        """Reject a working subdir outside the workspace, or an unservable tool."""
        _read_only_boolean(self.read_only_workspace)
        validate_read_only_workspace(self.environment.type, self.read_only_workspace)
        # A bare `working_subdir:` or `tools:` key in yaml decodes to None, not to
        # the field default, so neither the join below nor an environment has to
        # defend against it.
        if self.working_subdir is None:
            object.__setattr__(self, "working_subdir", "")
        if self.tools is None:
            object.__setattr__(self, "tools", [])
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
        validate_tool_capabilities(self.environment.type, self.tools)

    @property
    def working_dir(self) -> str:
        """The absolute sandbox directory the agent's commands run in.

        Derived rather than authored, so a prompt naming this directory and the
        environment running the agent there cannot disagree.
        """
        return posixpath.normpath(
            posixpath.join(DEFAULT_WORKSPACE_DIR, self.working_subdir)
        )
