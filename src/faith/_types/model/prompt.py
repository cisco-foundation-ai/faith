# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides `PromptFormatter` for formatting prompts for different model types."""

from typing import Callable

from faith._types.enums import CIEnum
from faith._types.record.model_response import ChatConversation


def _format_base_input(
    prompt: str,
    _system_prompt: str | None = None,
    response_leadin: str | None = None,
) -> str:
    """Format a prompt into a string for base input.

    The system prompt is ignored in this format, as base models typically do not
    use a system prompt in the same way chat models do.
    """
    return f"{prompt}\n{response_leadin or ''}"


def _format_chat_input(
    prompt: str,
    system_prompt: str | None = None,
    response_leadin: str | None = None,
) -> ChatConversation:
    """Format a prompt into a chat conversation message structure.

    This is used for chat models that expect a list of messages with roles in the
    OpenAI chat format.
    """
    return [
        msg
        for msg in [
            (
                {"role": "system", "content": system_prompt}
                if system_prompt is not None
                else None
            ),
            {"role": "user", "content": prompt},
            (
                {"role": "assistant", "content": response_leadin}
                if response_leadin is not None
                else None
            ),
        ]
        if msg is not None
    ]


class PromptFormatter(CIEnum):
    """Enum for different model prompt types."""

    BASE = (_format_base_input,)
    CHAT = (_format_chat_input,)

    @property
    def formatter(self) -> Callable[..., str | ChatConversation]:
        """Return the formatting function for this prompt type."""
        return self.value[0]

    def format(
        self,
        system_prompt: str | None,
        prompt: str,
        response_leadin: str | None,
    ) -> str | ChatConversation:
        """Format the prompt using the underlying formatter for the enum value.

        Args:
            system_prompt: Optional system prompt to include in the formatted output.
            prompt: The main prompt text to format.
            response_leadin: Optional lead-in text for the response.
        """
        return self.formatter(prompt, system_prompt, response_leadin)
