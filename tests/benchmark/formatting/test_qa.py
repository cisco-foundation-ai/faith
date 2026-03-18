# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._types.config.format import FormatConfig, InstructionsConfig, PromptConfig
from faith._types.records.prompt_record import PromptRecord
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.benchmark.formatting.qa import QAFormatter

_PROMPT_TEMPLATE = """{%- if instruction -%}
{{ instruction }}

{% endif -%}
{% if examples -%}
{% for example in examples -%}
{{ example.question }}
{{ example.answer }}

{% endfor -%}
{% endif -%}
{{ question }}"""

_FORMAT_CFG = FormatConfig(
    instructions=InstructionsConfig(
        system_prompt_template="System prompt for {{ subject }}",
        base_inst_template="Basic instruction template for {{ subject }}",
        chat_inst_template="Chat instruction template for {{ subject }}",
    ),
    prompt=PromptConfig(
        question_template="Question: {{ question }}",
        answer_template="Answer: {{ answer }}",
        prompt_template=_PROMPT_TEMPLATE,
    ),
)


def test_qa_formatter() -> None:
    # pylint: disable=protected-access
    format_cfg = _FORMAT_CFG

    formatter = QAFormatter(PromptFormatter.BASE, format_cfg=format_cfg)

    # Test instruction rendering with and without choices.
    assert (
        formatter._instruction(subject="Math") == "Basic instruction template for Math"
    )
    assert (
        formatter._instruction(choices=["A", "B"], subject="History")
        == "Basic instruction template for History"
    )

    # Test rendering a question and answer.
    assert (
        formatter._render_question("What is the capital of France?")
        == "Question: What is the capital of France?"
    )
    assert formatter.render_answer("Paris") == "Answer: Paris"

    # Test rendering a prompt.
    assert (
        formatter._render_prompt(
            instruction=None,
            examples=[],
            question="What is the capital of France?",
        )
        == "What is the capital of France?"
    )
    assert (
        formatter._render_prompt(
            instruction="Please answer the following question.",
            examples=[],
            question="What is the capital of France?",
        )
        == """Please answer the following question.

What is the capital of France?"""
    )
    assert (
        formatter._render_prompt(
            instruction="Please answer the following question.",
            examples=[
                PromptRecord(
                    benchmark_sample_index=1,
                    benchmark_sample_hash="hash123",
                    subject=None,
                    system_prompt=None,
                    instruction="Instruction",
                    question="What is the capital of Germany?",
                    choices={"A": "Berlin", "B": "Munich"},
                    label="A",
                    formatted_question="Question: What is the capital of Germany?",
                    formatted_answer="Answer: A",
                    question_prompt="Full question with context",
                    ancillary_data=None,
                )
            ],
            question="Question: What is the capital of France?",
        )
        == """Please answer the following question.

Question: What is the capital of Germany?
Answer: A

Question: What is the capital of France?"""
    )

    # Test rendering an example with no choices but an example.
    assert formatter.render_qa_record(
        index=1,
        sample_hash="hash123",
        raw_question="What is the capital of France?",
        raw_answer="Paris",
        examples=[
            PromptRecord(
                benchmark_sample_index=1,
                benchmark_sample_hash="hash654",
                subject="Geography",
                system_prompt="System prompt",
                instruction="Basic instruction template for Geography",
                question="What is the capital of Germany?",
                choices=None,
                label="Berlin",
                formatted_question="Question: What is the capital of Germany?",
                formatted_answer="Answer: Berlin",
                question_prompt="Basic instruction template for Geography\n\nQuestion: What is the capital of Germany?",
                ancillary_data=None,
            )
        ],
        subject="Geography",
    ) == PromptRecord(
        benchmark_sample_index=1,
        benchmark_sample_hash="hash123",
        subject="Geography",
        system_prompt="System prompt for Geography",
        instruction="Basic instruction template for Geography",
        question="What is the capital of France?",
        choices=None,
        label="Paris",
        formatted_question="Question: What is the capital of France?",
        formatted_answer="Answer: Paris",
        question_prompt="Basic instruction template for Geography\n\nQuestion: What is the capital of Germany?\nAnswer: Berlin\n\nQuestion: What is the capital of France?",
        ancillary_data=None,
    )

    # Test rendering an example with choices.
    assert formatter.render_qa_record(
        index=1,
        sample_hash="hash123",
        raw_question="What is the capital of France?",
        raw_answer="A",
        examples=[],
        choice_map={"A": "Paris", "B": "London"},
        subject="Geography",
    ) == PromptRecord(
        benchmark_sample_index=1,
        benchmark_sample_hash="hash123",
        subject="Geography",
        system_prompt="System prompt for Geography",
        instruction="Basic instruction template for Geography",
        question="What is the capital of France?",
        choices={"A": "Paris", "B": "London"},
        label="A",
        formatted_question="Question: What is the capital of France?",
        formatted_answer="Answer: A",
        question_prompt="Basic instruction template for Geography\n\nQuestion: What is the capital of France?",
        ancillary_data=None,
    )


def test_qa_formatter_render_conversation() -> None:
    format_cfg = FormatConfig(
        instructions=InstructionsConfig(
            system_prompt_template="System prompt",
            base_inst_template="Basic instruction template for {{ subject }}",
            chat_inst_template="Chat instruction template for {{ subject }}",
        ),
        prompt=PromptConfig(
            question_template="Question: {{ question }}",
            answer_template="Answer: {{ answer }}",
            prompt_template=_PROMPT_TEMPLATE,
        ),
    )

    base_formatter = QAFormatter(PromptFormatter.BASE, format_cfg=format_cfg)
    assert (
        base_formatter.render_conversation(
            PromptRecord(
                benchmark_sample_index=1,
                benchmark_sample_hash="hash123",
                subject="Geography",
                system_prompt="System prompt",
                instruction="Basic instruction template for Geography",
                question="What is the capital of France?",
                choices=None,
                label="Paris",
                formatted_question="Question: What is the capital of France?",
                formatted_answer="Answer: Paris",
                question_prompt="Basic instruction template for Geography\n\nQuestion: What is the capital of Germany?\nAnswer: Berlin\n\nQuestion: What is the capital of France?",
                ancillary_data=None,
            ),
            None,
        )
        == """Basic instruction template for Geography

Question: What is the capital of Germany?
Answer: Berlin

Question: What is the capital of France?
"""
    )

    chat_formatter = QAFormatter(PromptFormatter.CHAT, format_cfg=format_cfg)
    assert chat_formatter.render_conversation(
        PromptRecord(
            benchmark_sample_index=1,
            benchmark_sample_hash="hash123",
            subject="Geography",
            system_prompt="System prompt",
            instruction="Basic instruction template for Geography",
            question="What is the capital of France?",
            choices={"A": "Paris", "B": "London"},
            label="A",
            formatted_question="Question: What is the capital of France?",
            formatted_answer="Answer: A",
            question_prompt="Basic instruction template for Geography\n\nQuestion: What is the capital of France?",
            ancillary_data=None,
        ),
        "Answer:",
    ) == [
        {"role": "system", "content": "System prompt"},
        {
            "role": "user",
            "content": "Basic instruction template for Geography\n\nQuestion: What is the capital of France?",
        },
        {"role": "assistant", "content": "Answer:"},
    ]
