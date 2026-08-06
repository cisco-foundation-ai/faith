# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Tool base class's execute template.

``execute`` validates arguments and never raises; subclasses supply only
``_invoke``. ``missing_capabilities`` reports what the environment cannot serve.
"""

from typing import Any

import pytest

from faith.agentic.core.capabilities import CommandExecution, FileRead
from faith.agentic.core.environment import Environment
from faith.agentic.core.task import Task
from faith.agentic.core.tool import Tool

_ECHO_PARAMETERS = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "extra": {"type": "string"}},
    "required": ["text"],
}

# A hand-written schema may spell "nothing is required" as null rather than [].
_NULL_REQUIRED_PARAMETERS = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": None,
}

_MISSING_TEXT_ERROR = "Error: echo requires 'text' argument"
_NON_OBJECT_ERROR = "Error: echo requires a JSON object of arguments"


class _NoCapabilityEnvironment(Environment):
    """Environment satisfying no capability: lifecycle is all Environment requires."""

    def setup(self, task: Task) -> None:
        pass

    def teardown(self) -> None:
        pass


class _ShellEnvironment(_NoCapabilityEnvironment):
    """A backend with a shell but no file API, so it satisfies CommandExecution only."""

    def run_command(self, command: str) -> tuple[int, str]:
        """Return canned shell output."""
        return 0, f"ran {command}"


class _EchoTool(Tool):
    """Tool echoing its required 'text' argument, with 'extra' optional.

    Concrete because Tool is abstract; its parameters are settable so a test can
    vary the schema `execute` validates against without another subclass.
    """

    def __init__(self, parameters: dict | None = None) -> None:
        self._parameters = _ECHO_PARAMETERS if parameters is None else parameters

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo the text back."

    @property
    def parameters(self) -> dict:
        return self._parameters

    def _invoke(self, environment: Any, args: dict) -> str:
        extra = args.get("extra")
        return f"echo: {args['text']}{f' ({extra})' if extra else ''}"


class _ExplodingTool(_EchoTool):
    """Tool whose _invoke always raises, exercising the never-raise contract."""

    def _invoke(self, environment: Any, args: dict) -> str:
        raise RuntimeError("backend gone")


class _ShellTool(_EchoTool):
    """Tool requiring CommandExecution."""

    required_capabilities = (CommandExecution,)


class _ShellAndReadTool(_EchoTool):
    """Tool requiring both CommandExecution and FileRead."""

    required_capabilities = (CommandExecution, FileRead)


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"text": "hi"}, "echo: hi"),
        # Only 'required' keys are enforced, so an optional one flows through.
        ({"text": "hi", "extra": "e"}, "echo: hi (e)"),
        # '' is a supplied value; only a missing key is an error.
        ({"text": ""}, "echo: "),
        # A malformed call is an error string fed back to the model, never an exception.
        ({}, _MISSING_TEXT_ERROR),
        ({"extra": "e"}, _MISSING_TEXT_ERROR),
        # A model may emit a bare scalar or list, which must not escape execute().
        (5, _NON_OBJECT_ERROR),
        (None, _NON_OBJECT_ERROR),
        (["text"], _NON_OBJECT_ERROR),
        ("hi", _NON_OBJECT_ERROR),
    ],
)
def test_execute_validates_arguments_then_invokes(args: Any, expected: str) -> None:
    assert _EchoTool().execute(_NoCapabilityEnvironment(), args) == expected


def test_execute_turns_a_raising_invoke_into_an_error_string() -> None:
    assert _ExplodingTool().execute(_NoCapabilityEnvironment(), {"text": "hi"}) == (
        "Error executing echo: backend gone"
    )


def test_execute_treats_a_null_required_list_as_empty() -> None:
    """A null 'required' enforces nothing, rather than escaping execute()."""
    tool = _EchoTool(parameters=_NULL_REQUIRED_PARAMETERS)
    assert tool.execute(_NoCapabilityEnvironment(), {"text": "hi"}) == "echo: hi"


def test_a_tool_is_not_terminal_by_default() -> None:
    """A tool opts into terminality, and into carrying an answer, explicitly."""
    tool = _EchoTool()
    assert (tool.is_terminal, tool.answer_field) == (False, None)


@pytest.mark.parametrize(
    "tool,environment,expected",
    [
        # A tool needing nothing is served by the barest environment.
        (_EchoTool(), _NoCapabilityEnvironment(), ()),
        # Every unsatisfied capability is reported, in declaration order, not just
        # the first: the caller reports them together, so a truncated tuple would
        # hide work the environment still needs.
        (
            _ShellAndReadTool(),
            _NoCapabilityEnvironment(),
            (CommandExecution, FileRead),
        ),
        # Capabilities match structurally, so a partial environment is legal and
        # the ones it does satisfy drop out.
        (_ShellAndReadTool(), _ShellEnvironment(), (FileRead,)),
        (_ShellTool(), _ShellEnvironment(), ()),
    ],
)
def test_missing_capabilities_reports_the_unsatisfied_ones(
    tool: Tool, environment: Environment, expected: tuple[type, ...]
) -> None:
    """Test a tool reports exactly the capabilities its environment cannot serve."""
    assert tool.missing_capabilities(environment) == expected
