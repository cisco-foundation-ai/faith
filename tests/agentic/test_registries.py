# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for config-time tool/environment capability validation.

Both real environments satisfy every capability, so the rejection path is
exercised through fake registries holding a shell-only environment and a tool
that needs more than one capability. The check compares CLASSES, so nothing here
touches a container or a cluster.
"""

from unittest.mock import patch

import pytest

from faith.agentic.core.capabilities import FileRead, FileWrite
from faith.agentic.core.environment import Environment
from faith.agentic.core.task import Task
from faith.agentic.core.tool import Tool
from faith.agentic.registries import (
    AgenticEnvironmentType,
    AgenticToolType,
    _ConfiguredRegistry,
    _NoConfigRegistry,
    validate_tool_capabilities,
)
from faith.agentic.tools import BashExecTool, SubmitTool


class _ShellOnlyEnvironment(Environment):
    """An environment with a shell but no file API."""

    def setup(self, task: Task) -> None:
        pass

    def run_command(self, _command: str) -> tuple[int, str]:
        """Return canned shell output."""
        return 0, ""

    def teardown(self) -> None:
        pass


class _ReadTool(Tool):
    """A tool needing a capability the shell-only environment lacks."""

    required_capabilities = (FileRead,)

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return "Read a file."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def _invoke(self, environment, args):
        return "ok"


class _CopyTool(Tool):
    """A tool needing two capabilities, which no built-in tool does."""

    required_capabilities = (FileRead, FileWrite)

    @property
    def name(self) -> str:
        return "copy"

    @property
    def description(self) -> str:
        return "Copy a file."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def _invoke(self, environment, args):
        return "ok"


class _FakeEnvType(_ConfiguredRegistry):
    SHELL_ONLY = (_ShellOnlyEnvironment,)


class _FakeToolType(_NoConfigRegistry):
    BASH_EXEC = (BashExecTool,)
    SUBMIT = (SubmitTool,)
    READ = (_ReadTool,)
    COPY = (_CopyTool,)


def test_every_real_environment_serves_every_builtin_tool() -> None:
    """Docker and Kubernetes both satisfy every built-in tool's capabilities.

    This is what lets a benchmark.yaml pair any built-in tool with either
    environment, and it checks the real classes: validation resolves them out of
    the production registry rather than being told which class to look at.
    """
    tool_names = [str(tool) for tool in AgenticToolType]
    for env_type in AgenticEnvironmentType:
        validate_tool_capabilities(str(env_type), tool_names)


def test_no_tools_is_valid() -> None:
    """A benchmark may enable no tools at all; the loop then just has nothing to call."""
    validate_tool_capabilities("docker", [])


@pytest.mark.parametrize(
    ("environment_type", "tool_names", "expected_message"),
    [
        ("nope", [], "'nope' is not a valid AgenticEnvironmentType"),
        ("docker", ["nope"], "'nope' is not a valid AgenticToolType"),
    ],
    ids=["unregistered_environment", "unregistered_tool"],
)
def test_an_unregistered_name_is_rejected(
    environment_type: str, tool_names: list[str], expected_message: str
) -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_tool_capabilities(environment_type, tool_names)
    assert str(excinfo.value) == expected_message


@patch("faith.agentic.registries.AgenticEnvironmentType", _FakeEnvType)
def test_tools_the_environment_can_serve_are_accepted() -> None:
    """bash_exec needs the shell the fake has; submit needs no capability at all."""
    validate_tool_capabilities("shell_only", ["bash_exec", "submit"])


@pytest.mark.parametrize(
    ("tool_names", "expected_message"),
    [
        (
            ["bash_exec", "read"],
            "Environment 'shell_only' does not support tool 'read' "
            "(requires FileRead). Configured tools: bash_exec, read.",
        ),
        (
            ["read", "submit"],
            "Environment 'shell_only' does not support tool 'read' "
            "(requires FileRead). Configured tools: read, submit.",
        ),
    ],
    ids=["first_tool_serviceable", "offender_first"],
)
@patch("faith.agentic.registries.AgenticToolType", _FakeToolType)
@patch("faith.agentic.registries.AgenticEnvironmentType", _FakeEnvType)
def test_a_tool_the_environment_cannot_serve_is_rejected(
    tool_names: list[str], expected_message: str
) -> None:
    """The message carries the whole diagnosis: a benchmark author reads only it."""
    with pytest.raises(ValueError) as excinfo:
        validate_tool_capabilities("shell_only", tool_names)
    assert str(excinfo.value) == expected_message


@patch("faith.agentic.registries.AgenticToolType", _FakeToolType)
@patch("faith.agentic.registries.AgenticEnvironmentType", _FakeEnvType)
def test_every_capability_a_tool_is_missing_is_named() -> None:
    """One missing capability per line would hide the second from the author."""
    with pytest.raises(ValueError) as excinfo:
        validate_tool_capabilities("shell_only", ["copy"])
    assert str(excinfo.value) == (
        "Environment 'shell_only' does not support tool 'copy' "
        "(requires FileRead, FileWrite). Configured tools: copy."
    )
