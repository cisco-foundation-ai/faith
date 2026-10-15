# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ToolRegistry.

The registry keys tools by name, advertises their function-calling schemas, and
routes calls. A tool honoring ``Tool.execute``'s never-raise contract yields a
string; one that raises anyway is wrapped in ToolExecutionError, which the agent
loop distinguishes from an unknown tool name.
"""

from typing import Any, cast

import pytest

from faith.agentic.core import (
    Environment,
    Tool,
    ToolArgumentError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
)

# FakeTool ignores its environment, and the registry only forwards it.
_NO_ENV = cast(Environment, None)

_FAKE_PARAMETERS = {
    "type": "object",
    "properties": {"arg": {"type": "string", "description": "Test argument"}},
    "required": ["arg"],
}


class _FakeTool(Tool):
    """Fake tool echoing the arguments it was invoked with."""

    def __init__(self, name: str, description: str = "Fake tool") -> None:
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return _FAKE_PARAMETERS

    def _invoke(self, environment: Any, args: dict) -> str:
        return f"FakeTool({self._name}) executed with {args}"


class _BrokenTool:
    """A tool whose `execute` raises, which `Tool` itself forbids.

    Deliberately not a Tool subclass: `Tool.execute` is final, so the only way
    to reach the registry's wrapping is a duck-typed stand-in cast at the
    boundary. Models the real hazard of a third-party tool that does not honor
    the never-raise contract.
    """

    def __init__(self, name: str, error: Exception) -> None:
        self.name = name
        self.description = "Broken tool"
        self.parameters: dict = {"type": "object", "properties": {}}
        self._error = error

    def execute(self, environment: Any, args: dict) -> str:
        """Raise instead of returning."""
        raise self._error

    def validate_args(self, _args: dict) -> str | None:
        """Accept arguments so the test reaches execution."""
        return None


def _broken_registry(name: str, error: Exception) -> ToolRegistry:
    """Registry serving a single tool that raises `error` from execute."""
    return ToolRegistry([cast(Tool, _BrokenTool(name, error))])


def test_a_registered_tool_is_listed_and_retrievable() -> None:
    tool = _FakeTool("test_tool")
    registry = ToolRegistry([tool])

    assert registry.list_tools() == ["test_tool"]
    assert registry.get_tool("test_tool") is tool


def test_tools_are_listed_in_registration_order() -> None:
    """The advertised order is the registered order, and an empty registry is legal."""
    assert not ToolRegistry().list_tools()
    assert ToolRegistry(
        [_FakeTool("tool_a"), _FakeTool("tool_b"), _FakeTool("tool_c")]
    ).list_tools() == ["tool_a", "tool_b", "tool_c"]


def test_duplicate_tool_name_raises() -> None:
    """Two tools sharing a name would make one unreachable by that key."""
    with pytest.raises(ValueError, match="already registered"):
        ToolRegistry([_FakeTool("duplicate"), _FakeTool("duplicate")])


def test_get_tool_not_found_names_the_available_tools() -> None:
    """The error is the caller's only clue, so it lists what it could have used."""
    registry = ToolRegistry([_FakeTool("tool_a"), _FakeTool("tool_b")])

    with pytest.raises(ToolNotFoundError) as exc_info:
        registry.get_tool("nonexistent")

    assert str(exc_info.value) == (
        "Tool 'nonexistent' not found. Available tools: tool_a, tool_b"
    )


def test_get_function_schemas() -> None:
    registry = ToolRegistry(
        [
            _FakeTool("bash_exec", "Execute bash command"),
            _FakeTool("file_read", "Read file contents"),
        ]
    )

    assert registry.get_function_schemas() == [
        {
            "type": "function",
            "function": {
                "name": "bash_exec",
                "description": "Execute bash command",
                "parameters": _FAKE_PARAMETERS,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read file contents",
                "parameters": _FAKE_PARAMETERS,
            },
        },
    ]


def test_function_schemas_advertise_the_registered_name() -> None:
    """A tool renamed after registration is still resolvable under what we advertise.

    The dict key, not the tool's current `name`, is what execute_tool resolves.
    """

    class RenamableTool(_FakeTool):
        """Tool whose name can change after it is registered."""

        def rename(self, name: str) -> None:
            """Change the name this tool reports."""
            self._name = name

    tool = RenamableTool("before")
    registry = ToolRegistry([tool])
    tool.rename("after")

    advertised = registry.get_function_schemas()[0]["function"]["name"]

    assert advertised == "before"
    assert registry.get_tool(advertised) is tool


def test_mutating_an_advertised_schema_leaves_the_tool_contract_alone() -> None:
    """The advertised `parameters` is a copy, not the dict execute validates against.

    The schemas are handed to model-API code, so a consumer editing one would
    otherwise rewrite what the tool accepts for the rest of its life.
    """
    registry = ToolRegistry([_FakeTool("test_tool")])

    registry.get_function_schemas()[0]["function"]["parameters"]["required"].append(
        "injected"
    )

    assert registry.get_function_schemas()[0]["function"]["parameters"] == (
        _FAKE_PARAMETERS
    )
    assert registry.execute_tool("test_tool", {"arg": "value"}, _NO_ENV) == (
        "FakeTool(test_tool) executed with {'arg': 'value'}"
    )


def test_execute_tool_routes_to_the_named_tool() -> None:
    registry = ToolRegistry([_FakeTool("test_tool")])

    assert registry.execute_tool("test_tool", {"arg": "value"}, _NO_ENV) == (
        "FakeTool(test_tool) executed with {'arg': 'value'}"
    )


def test_argument_rejection_is_distinct_from_execution_failure() -> None:
    registry = ToolRegistry([_FakeTool("test_tool")])
    with pytest.raises(ToolArgumentError, match="requires 'arg' argument"):
        registry.execute_tool("test_tool", {}, _NO_ENV)


def test_execute_tool_rejects_an_unregistered_name() -> None:
    with pytest.raises(ToolNotFoundError, match="not found"):
        ToolRegistry().execute_tool("nonexistent", {}, _NO_ENV)


def test_execute_tool_returns_a_conforming_tools_error_string() -> None:
    """A tool honoring the never-raise contract yields a string, not an exception."""

    class RaisingInvokeTool(_FakeTool):
        """Tool whose _invoke raises, which Tool.execute converts to a string."""

        def _invoke(self, environment: Any, args: dict) -> str:
            raise RuntimeError("backend gone")

    registry = ToolRegistry([RaisingInvokeTool("flaky")])

    assert registry.execute_tool("flaky", {"arg": "v"}, _NO_ENV) == (
        "Error executing flaky: backend gone"
    )


@pytest.mark.parametrize(
    "error,expected",
    [
        (
            RuntimeError("boom"),
            "Tool 'failing' failed during execution: RuntimeError: boom",
        ),
        # The message is all the caller gets, so a message-less error still says
        # what it was.
        (TimeoutError(), "Tool 'failing' failed during execution: TimeoutError: "),
        # A ValueError from inside a tool must not be mistaken for an unknown
        # name: the agent loop catches ToolNotFoundError separately.
        (
            ValueError("bad argument"),
            "Tool 'failing' failed during execution: ValueError: bad argument",
        ),
    ],
    ids=["runtime_error", "no_message", "value_error"],
)
def test_a_raising_tool_is_wrapped_in_tool_execution_error(
    error: Exception, expected: str
) -> None:
    with pytest.raises(ToolExecutionError) as exc_info:
        _broken_registry("failing", error).execute_tool("failing", {}, _NO_ENV)

    assert str(exc_info.value) == expected
    assert exc_info.value.__cause__ is error
    assert not isinstance(exc_info.value, ToolNotFoundError)


def test_tool_not_found_is_a_value_error() -> None:
    """Existing `except ValueError` handlers must still catch an unknown tool."""
    assert issubclass(ToolNotFoundError, ValueError)
