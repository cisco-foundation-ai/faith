# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark formatting configuration types for instructions and prompts."""

from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin


@dataclass(frozen=True)
class InstructionsConfig(DataClassJsonMixin):
    """Configuration for benchmark instruction templates."""

    system_prompt_template: str | None = None
    base_inst_template: str | None = None
    chat_inst_template: str | None = None


@dataclass(frozen=True)
class PromptConfig(DataClassJsonMixin):
    """Configuration for benchmark prompt templates."""

    question_template: str | None = None
    answer_template: str | None = None
    prompt_template: str | None = None


@dataclass(frozen=True)
class FormatConfig(DataClassJsonMixin):
    """Configuration for benchmark formatting (instructions + prompts)."""

    instructions: InstructionsConfig = field(default_factory=InstructionsConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
