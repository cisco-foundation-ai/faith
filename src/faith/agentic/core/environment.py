# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Base Environment abstraction for agentic evaluation."""

from abc import ABC, abstractmethod

from faith.agentic.core.task import Task

# The absolute directory inside every sandbox that a task's data is unpacked
# into, and that a workspace-relative path resolves against. A constant rather
# than a config key: it is the boundary the path guard enforces, the destination
# the backends upload to, and what an agent image is built to provide, so a
# benchmark that could move it could defeat all three.
DEFAULT_WORKSPACE_DIR = "/workspace"


class EnvironmentUnavailableError(RuntimeError):
    """Raised when the sandbox is gone or unreachable, not when work in it failed.

    A command exiting non-zero is an ordinary result; this is a terminated pod, a
    dead container, or a lost connection. The loop rebuilds and retries rather than
    charging the call, so the agent is never billed for a cluster hiccup.
    """


class Environment(ABC):
    """Base class for agent execution environments.

    An environment is an isolated sandbox. Only its lifecycle is required here;
    what it can *do* is expressed by the capability protocols it satisfies (see
    ``core.capabilities``).

    An implementation whose sandbox can die mid-task raises
    EnvironmentUnavailableError from its capability methods, so the loop can tell
    infrastructure failure from a command the agent got wrong.
    """

    @abstractmethod
    def setup(self, task: Task) -> None:
        """Set up the environment for a task.

        Args:
            task: Task to set up for. An environment may read its `id`,
                `data_path`, and `environment_config`; `prompt` and
                `ground_truth` are the loop's and the grader's concern, not the
                sandbox's.

        Raises:
            RuntimeError: If environment setup fails.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Clean up environment resources. Must be idempotent."""
