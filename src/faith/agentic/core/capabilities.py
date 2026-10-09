# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Optional environment capabilities required by tools or benchmark policy.

``Environment`` itself covers only lifecycle; what an environment can *do* is
expressed by the capabilities it satisfies. These are structural Protocols, so an
environment declares one by implementing its method and a tool names what it
needs in ``Tool.required_capabilities``.

Every capability must declare METHODS ONLY: config-time validation calls
``issubclass`` against these, which a Protocol carrying a data member rejects
with a TypeError.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ReadOnlyWorkspace(Protocol):
    """Can irreversibly protect task inputs before the agent starts."""

    def seal_workspace(self) -> None:
        """Protect the workspace from shell and file-API writes, or raise.

        Setup must call this when read_only_workspace is required, including
        when rebuilding an environment after an infrastructure failure.
        """


@runtime_checkable
class CommandExecution(Protocol):
    """Runs shell commands. Satisfied by any shell-backed sandbox."""

    def run_command(self, command: str) -> tuple[int, str]:
        """Run a shell command.

        Args:
            command: The shell command to run.

        Returns:
            (exit_code, combined_output), unformatted.

        Raises:
            RuntimeError: If the command cannot be run.
        """


@runtime_checkable
class FileRead(Protocol):
    """Reads workspace files as text."""

    def read_file(self, path: str) -> str:
        """Read a workspace file as text.

        Args:
            path: Sandbox path, relative to the environment's workspace root or
                absolute within it. Never a host path.

        Returns:
            The file contents as text.

        Raises:
            ValueError: If the path escapes the workspace.
            RuntimeError: If the read fails.
        """


@runtime_checkable
class FileWrite(Protocol):
    """Writes workspace files."""

    def write_file(self, path: str, content: str) -> None:
        """Write text to a workspace file, creating or overwriting it.

        Args:
            path: Sandbox path, relative to the environment's workspace root or
                absolute within it. Never a host path.
            content: Text to write.

        Raises:
            ValueError: If the path escapes the workspace.
            RuntimeError: If the write fails.
        """
