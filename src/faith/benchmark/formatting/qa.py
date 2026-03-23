# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""This module defines the QA example formatter, which constructs question-answer pairs."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from jinja2 import Template

from faith._types.config.format import FormatConfig
from faith._types.model.prompt import PromptFormatter
from faith._types.record.model_response import ChatConversation
from faith._types.record.prompt import PromptRecord


@dataclass(frozen=True)
class _QA:
    """Container for a question-answer pair.

    This is used internally by the `QAFormatter` to render question-answer pairs
    in the prompt.
    """

    question: str
    answer: str | None


def _opt_template(template_str: str | None) -> Template | None:
    """Helper function to convert a template string to a Jinja2 Template or None."""
    return Template(template_str) if template_str is not None else None


class QAFormatter:
    """A configurable formatter that constructs question-answer pairs for a benchmark."""

    def __init__(
        self, prompt_format: PromptFormatter, format_cfg: FormatConfig
    ) -> None:
        """Configures the formatter according to the given configs."""
        self._prompt_format = prompt_format
        self._system_prompt_template = _opt_template(
            format_cfg.instructions.system_prompt_template
        )
        inst_tmpl: str | None = None
        if prompt_format == PromptFormatter.BASE:
            inst_tmpl = format_cfg.instructions.base_inst_template
        elif prompt_format == PromptFormatter.CHAT:
            inst_tmpl = format_cfg.instructions.chat_inst_template
        self._inst_template = _opt_template(inst_tmpl)
        self._question_template = _opt_template(format_cfg.prompt.question_template)
        self._answer_template = _opt_template(format_cfg.prompt.answer_template)
        self._prompt_template = _opt_template(format_cfg.prompt.prompt_template)

    def _render_system_prompt(self, subject: str | None = None) -> str | None:
        if self._system_prompt_template is None:
            return None
        return self._system_prompt_template.render(subject=subject)

    def _instruction(
        self, choices: Sequence[str] | None = None, subject: str | None = None
    ) -> str | None:
        """Fetch the instruction for this benchmark with the given subject."""
        if self._inst_template is None:
            return None
        return self._inst_template.render(choices=choices, subject=subject)

    def _render_prompt(
        self, instruction: str | None, examples: Sequence[PromptRecord], question: str
    ) -> str:
        """Renders the prompt using the prompt template with parameters instruction, examples, and question."""
        assert (
            self._prompt_template is not None
        ), "Prompt template is not defined for this formatter."
        return self._prompt_template.render(
            instruction=instruction,
            examples=[
                _QA(question=ex.formatted_question, answer=ex.formatted_answer)
                for ex in examples
            ],
            question=question,
        )

    def _render_question(
        self, question: str, choice_map: dict[str, str] | None = None
    ) -> str:
        """Renders the question using the question template and choice map."""
        assert (
            self._question_template is not None
        ), "Question template is not defined for this formatter."
        return self._question_template.render(question=question, choice_map=choice_map)

    def render_answer(self, answer: str | None) -> str | None:
        """Renders the answer using the answer template."""
        assert (
            self._answer_template is not None
        ), "Answer template is not defined for this formatter."
        if answer is None:
            return None
        return self._answer_template.render(answer=answer)

    def render_qa_record(
        self,
        index: int,
        sample_hash: str,
        raw_question: str,
        raw_answer: str | None,
        examples: Sequence[PromptRecord] | None = None,
        choice_map: dict[str, str] | None = None,
        subject: str | None = None,
        ancillary_data: dict[str, Any] | None = None,
    ) -> PromptRecord:
        """Renders an example question-answer pair."""
        formatted_question = self._render_question(raw_question, choice_map)
        formatted_answer = self.render_answer(raw_answer)
        choice_list = sorted(choice_map.keys()) if choice_map else None
        instruction = self._instruction(choices=choice_list, subject=subject)
        question_prompt = self._render_prompt(
            instruction=instruction,
            examples=examples or [],
            question=formatted_question,
        )
        return PromptRecord(
            benchmark_sample_index=index,
            benchmark_sample_hash=sample_hash,
            subject=subject,
            system_prompt=self._render_system_prompt(subject=subject),
            instruction=instruction,
            question=raw_question,
            choices=choice_map,
            label=raw_answer,
            formatted_question=formatted_question,
            formatted_answer=formatted_answer,
            question_prompt=question_prompt,
            ancillary_data=ancillary_data,
        )

    def render_conversation(
        self, record: PromptRecord, answer_leadin: str | None
    ) -> str | ChatConversation:
        """Format the prompt for the given PromptRecord."""
        return self._prompt_format.format(
            system_prompt=record.system_prompt,
            prompt=record.question_prompt,
            response_leadin=answer_leadin,
        )
