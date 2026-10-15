# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tool registry for managing agent tools."""

from collections.abc import Iterable
from copy import deepcopy

from faith.agentic.core.environment import Environment, EnvironmentUnavailableError
from faith.agentic.core.tool import Tool


class ToolArgumentError(ValueError):
    """Raised when arguments are rejected before a tool executes."""


class ToolExecutionError(Exception):
    """Raised when a registered tool fails during execution.

    Wraps the tool's underlying exception (kept as ``__cause__``) so the agent
    loop can catch tool failures narrowly and feed them back to the model,
    distinct from a terminal loop error.
    """


class ToolNotFoundError(ValueError):
    """Raised when a tool name is not registered.

    Subclasses ValueError so existing handlers still catch it, but lets the agent
    loop tell an unknown tool name from an unrelated ValueError raised while
    running one.
    """


class ToolRegistry:
    """An immutable set of agent tools, keyed by name.

    Handles schema generation for function calling and routes tool calls to the
    appropriate tool implementation. The set is fixed at construction and each
    tool's schema is derived once, so the generated schemas never change.
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        """Initialize the registry with the tools it will serve.

        Args:
            tools: Tools to serve, each keyed by its own `name`.

        Raises:
            ValueError: If two tools share a name.
        """
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(
                    f"Tool '{tool.name}' is already registered. "
                    f"Each tool must have a unique name."
                )
            self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool:
        """Get a registered tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance

        Raises:
            ToolNotFoundError: If tool is not registered
        """
        if name not in self._tools:
            available = ", ".join(self._tools.keys())
            raise ToolNotFoundError(
                f"Tool '{name}' not found. Available tools: {available}"
            )
        return self._tools[name]

    def get_function_schemas(self) -> list[dict]:
        """Generate function calling schemas for all registered tools.

        Returns:
            List of OpenAI function calling schema dicts, each safe to mutate:
            `parameters` is copied rather than shared with the tool that
            validates calls against it.
        """
        return [
            {
                "type": "function",
                # The dict key overrides the schema's name: only the key is
                # guaranteed to be what execute_tool can resolve.
                "function": {
                    **tool.schema,
                    "name": name,
                    "parameters": deepcopy(tool.parameters),
                },
            }
            for name, tool in self._tools.items()
        ]

    def execute_tool(self, tool_name: str, args: dict, environment: Environment) -> str:
        """Execute a tool call.

        Args:
            tool_name: Name of tool to execute
            args: Tool arguments
            environment: Environment to execute tool in

        Returns:
            Tool execution result as a string

        Raises:
            ToolNotFoundError: If the tool is not found
            ToolArgumentError: If argument validation rejects the call
            EnvironmentUnavailableError: If the sandbox is gone or unreachable
            ToolExecutionError: If the tool raises during execution
        """
        tool = self.get_tool(tool_name)
        if (error := tool.validate_args(args)) is not None:
            raise ToolArgumentError(error)
        try:
            return tool.execute(environment, args)
        except EnvironmentUnavailableError:
            raise
        except Exception as e:
            raise ToolExecutionError(
                # The type as well as the message: only this string reaches the
                # model, and a bare `raise TimeoutError()` has no message.
                f"Tool '{tool_name}' failed during execution: {type(e).__name__}: {e}"
            ) from e

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())
