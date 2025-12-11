# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""This module defines the QA example formatter, which constructs question-answer pairs."""

from dataclasses import dataclass
from typing import Any, Sequence

from dataclasses_json import DataClassJsonMixin
from jinja2 import Template

from faith._internal.algo.hash import dict_sha256
from faith.benchmark.formatting.prompt import ChatConversation, PromptFormatter


@dataclass
class _QA:
    """Container for a question-answer pair.

    This is used internally by the `QAFormatter` to render question-answer pairs
    in the prompt.
    """

    question: str
    answer: str


@dataclass(frozen=True)
class QARecord(DataClassJsonMixin):
    """Base class for benchmark examples."""

    # Metadata about the benchmark sample.
    benchmark_sample_index: int
    benchmark_sample_hash: str
    subject: str | None

    # Components that make up the question.
    system_prompt: str | None
    instruction: str | None
    question: str
    choices: dict[str, str] | None  # Maps symbols (e.g., 'A', 'B') to their choice.
    label: str  # aka the "answer" or "ground truth".

    # Formatted question and answer.
    formatted_question: str
    formatted_answer: str

    # The full question that is passed to the model.
    question_prompt: str

    # Any additional data associated with this example that is stored alongside it
    # for context or as part of subsequent metric computations.
    ancillary_data: dict[str, Any] | None

    def sha256(self) -> str:
        """Compute the SHA-256 hash of this example."""
        return dict_sha256(self.to_dict())


class QAFormatter:
    """A configurable formatter that constructs question-answer pairs for a benchmark."""

    def __init__(self, prompt_format: PromptFormatter, format_cfg: dict[str, Any]):
        """Configures the formatter according to the given configs."""
        self._prompt_format = prompt_format
        inst_cfg = format_cfg.get("instructions", {})
        prompt_cfg = format_cfg.get("prompt", {})
        self._system_prompt = inst_cfg.get("system_prompt", None)
        inst_tmpl: str | None = None
        if prompt_format == PromptFormatter.BASE:
            inst_tmpl = inst_cfg.get("base_inst_template", None)
        elif prompt_format == PromptFormatter.CHAT:
            inst_tmpl = inst_cfg.get("chat_inst_template", None)
        self._inst_template = Template(inst_tmpl) if inst_tmpl is not None else None
        self._question_template = Template(prompt_cfg["question_template"])
        self._answer_template = Template(prompt_cfg["answer_template"])
        self._prompt_template = Template(prompt_cfg["prompt_template"])

    def _instruction(
        self, choices: Sequence[str] | None = None, subject: str | None = None
    ) -> str | None:
        """Fetch the instruction for this benchmark with the given subject."""
        if self._inst_template is None:
            return None
        return self._inst_template.render(choices=choices, subject=subject)

    def _render_prompt(
        self, instruction: str | None, examples: Sequence[QARecord], question: str
    ) -> str:
        """Renders the prompt using the prompt template with parameters instruction, examples, and question."""
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
        return self._question_template.render(question=question, choice_map=choice_map)

    def render_answer(self, answer: str) -> str:
        """Renders the answer using the answer template."""
        return self._answer_template.render(answer=answer)

    def render_qa_record(
        self,
        index: int,
        sample_hash: str,
        raw_question: str,
        raw_answer: str,
        examples: Sequence[QARecord] | None = None,
        choice_map: dict[str, str] | None = None,
        subject: str | None = None,
        ancillary_data: dict[str, Any] | None = None,
    ) -> QARecord:
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
        return QARecord(
            benchmark_sample_index=index,
            benchmark_sample_hash=sample_hash,
            subject=subject,
            system_prompt=self._system_prompt,
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
        self, record: QARecord, answer_leadin: str | None
    ) -> str | ChatConversation:
        """Format the prompt for the given QARecord."""
        return self._prompt_format.format(
            system_prompt=record.system_prompt,
            prompt=record.question_prompt,
            response_leadin=answer_leadin,
        )
