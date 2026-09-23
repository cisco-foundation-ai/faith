# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Submit answer tool."""

from typing import Any

from faith.agentic.core.tool import Tool


class SubmitTool(Tool):
    """Terminal tool: the agent calls it to submit a final answer and end the run.

    The loop reads the answer from ``answer_field`` and stops, so this tool needs
    no environment capability and only echoes a confirmation.
    """

    @property
    def name(self) -> str:
        return "submit"

    @property
    def is_terminal(self) -> bool:
        return True

    @property
    def answer_field(self) -> str | None:
        return "answer"

    @property
    def description(self) -> str:
        return (
            "Submit your final answer to the task. "
            "Call this when you have completed the task and have an answer. "
            "This will end the task execution."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your final answer to the task",
                }
            },
            "required": ["answer"],
        }

    def _invoke(self, environment: Any, args: dict) -> str:
        """Confirm the submission. The loop, not the environment, ends the run."""
        return f"Answer submitted: {args['answer']}"
