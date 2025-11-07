# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith.benchmark.formatting.prompt import PromptFormatter


def test_prompt_formatter_str() -> None:
    """Test the __str__ method of PromptFormatter enum."""
    assert str(PromptFormatter.BASE) == "base"
    assert str(PromptFormatter.CHAT) == "chat"


def test_prompt_formatter_from_string() -> None:
    """Test the from_string method of PromptFormatter enum."""
    assert PromptFormatter.from_string("base") == PromptFormatter.BASE
    assert PromptFormatter.from_string("chat") == PromptFormatter.CHAT

    with pytest.raises(ValueError, match="Unknown prompt formatter: invalid"):
        PromptFormatter.from_string("invalid")


def test_prompt_formatter_base_format() -> None:
    """Test the base prompt formatting."""
    assert (
        PromptFormatter.BASE.format(
            system_prompt=None,
            prompt="Hello, world!",
            response_leadin=None,
        )
        == "Hello, world!\n"
    )
    assert (
        PromptFormatter.BASE.format(
            system_prompt="This is a system message.",
            prompt="Hello, world!",
            response_leadin="Response:",
        )
        == "Hello, world!\nResponse:"
    )


def test_prompt_formatter_chat_format() -> None:
    """Test the chat prompt formatting."""
    assert PromptFormatter.CHAT.format(
        system_prompt=None,
        prompt="Hello, world!",
        response_leadin=None,
    ) == [{"role": "user", "content": "Hello, world!"}]
    assert PromptFormatter.CHAT.format(
        system_prompt="This is a system message.",
        prompt="Hello, world!",
        response_leadin="Response:",
    ) == [
        {"role": "system", "content": "This is a system message."},
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "Response:"},
    ]
