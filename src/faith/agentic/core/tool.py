# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Base Tool abstraction for agentic evaluation."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, ClassVar, final

from faith.agentic.core.environment import Environment


class Tool(ABC):
    """Base class for agent tools.

    A subclass declares its function-calling schema (name / description /
    parameters) and implements ``_invoke``; ``execute`` adds argument validation
    and the never-raise contract.
    """

    # Capabilities (see core.capabilities) _invoke uses; checked before the run,
    # not per call.
    required_capabilities: ClassVar[tuple[type, ...]] = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name used in function calling.

        Returns:
            Tool name (e.g., "bash_exec", "submit")
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does.

        Returns:
            Tool description shown to the model
        """

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for tool parameters.

        Returns:
            JSON Schema dict describing tool parameters. ``execute`` validates
            calls against its ``required`` list.
        """

    @cached_property
    def schema(self) -> dict[str, Any]:
        """Function-calling schema for this tool, derived once per instance.

        Read by both the registry (what the model is told) and ``_validate_args``
        (what a call is checked against), so the two cannot diverge even if a
        subclass computes its schema properties.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @property
    def is_terminal(self) -> bool:
        """Whether calling this tool ends the run (submit, done, abstain, ...)."""
        return False

    @property
    def answer_field(self) -> str | None:
        """Which argument key carries this terminal tool's answer.

        The loop reads ``args[answer_field]``, treating a missing key as a
        botched submission. None, the default, means the call itself is the
        answer, so no key is read (e.g. an abstain tool taking no arguments): a
        terminal tool carrying an answer must name the key it arrives under.
        Ignored unless ``is_terminal``.
        """
        return None

    @final
    def execute(self, environment: Environment, args: dict) -> str:
        """Validate the arguments, then invoke the tool, never raising.

        Tool behavior belongs in ``_invoke``.

        Args:
            environment: Environment whose capabilities the tool composes.
            args: Tool arguments from the model's tool call.

        Returns:
            The tool result, or an error string fed back to the model.
        """
        if (error := self._validate_args(args)) is not None:
            return error
        try:
            return self._invoke(environment, args)
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error executing {self.name}: {e}"

    def missing_capabilities(self, environment: Environment) -> tuple[type, ...]:
        """Required capabilities this environment does not satisfy, if any.

        Args:
            environment: Environment to check.

        Returns:
            The unsatisfied capabilities, empty if the environment can serve
            this tool.
        """
        return tuple(
            cap
            for cap in self.required_capabilities
            if not isinstance(environment, cap)
        )

    def _validate_args(self, args: dict) -> str | None:
        """Error string for a non-object args or the first missing required key, else None."""
        if not isinstance(args, dict):
            return f"Error: {self.name} requires a JSON object of arguments"
        for key in self.schema["parameters"].get("required") or []:
            if key not in args:
                return f"Error: {self.name} requires '{key}' argument"
        return None

    @abstractmethod
    def _invoke(self, environment: Any, args: dict) -> str:
        """Perform the tool's action; may raise, as ``execute`` catches.

        Args:
            environment: Environment to act in, guaranteed to satisfy
                ``required_capabilities``. Typed Any so an override can narrow it
                to the capability protocols it uses.
            args: Validated tool arguments.

        Returns:
            The tool result, JSON-encoded if structured.
        """
