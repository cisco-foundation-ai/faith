# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from faith._internal.io.json import read_json_file, write_as_json
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.benchmark.formatting.qa import QAFormatter, QARecord


def test_qa_record_sha256() -> None:
    record = QARecord(
        benchmark_sample_index=6,
        benchmark_sample_hash="ffffbbbb",
        subject="analysis",
        system_prompt=None,
        instruction="Please evaluate the following limit.",
        question="What is the limit of sin(x)/x as x approaches 0?",
        choices=None,
        label="0",
        formatted_question="Question: What is the limit of sin(x)/x as x approaches 0?",
        formatted_answer="Answer: 0",
        question_prompt="Please evaluate the following limit.\n\nQuestion: What is the limit of sin(x)/x as x approaches 0?",
    )
    assert (
        record.sha256()
        == "735c2c600a48d2e20bfe6d3f43157253b68c69118bb9298dde889fbf6c59b029"
    )


def test_qa_record_to_dict() -> None:
    record = QARecord(
        benchmark_sample_index=6,
        benchmark_sample_hash="ffffbbbb",
        subject="analysis",
        system_prompt=None,
        instruction="Please evaluate the following limit.",
        question="What is the limit of sin(x)/x as x approaches 0?",
        choices=None,
        label="0",
        formatted_question="Question: What is the limit of sin(x)/x as x approaches 0?",
        formatted_answer="Answer: 0",
        question_prompt="Please evaluate the following limit.\n\nQuestion: What is the limit of sin(x)/x as x approaches 0?",
    )
    assert record.to_dict() == {
        "benchmark_sample_index": 6,
        "benchmark_sample_hash": "ffffbbbb",
        "subject": "analysis",
        "system_prompt": None,
        "instruction": "Please evaluate the following limit.",
        "question": "What is the limit of sin(x)/x as x approaches 0?",
        "choices": None,
        "label": "0",
        "formatted_question": "Question: What is the limit of sin(x)/x as x approaches 0?",
        "formatted_answer": "Answer: 0",
        "question_prompt": "Please evaluate the following limit.\n\nQuestion: What is the limit of sin(x)/x as x approaches 0?",
    }


@pytest.mark.parametrize(
    "record",
    [
        QARecord(
            benchmark_sample_index=7,
            benchmark_sample_hash="aaaaaaa",
            subject=None,
            system_prompt="Behave as the wind behaves.",
            instruction="Remember us - if at all - not as lost violent souls",
            question="What falls between the conception and the creation?",
            choices={"A": "The Kingdom", "B": "The Existence", "C": "The Shadow"},
            label="C",
            formatted_question="Formatted question",
            formatted_answer="Formatted answer",
            question_prompt="Full question with context",
        ),
        QARecord(
            benchmark_sample_index=1,
            benchmark_sample_hash="hash123",
            subject=None,
            system_prompt="System prompt",
            instruction="Instruction",
            question="What is the capital of France?",
            choices={"A": "Paris", "B": "London"},
            label="A",
            formatted_question="Formatted question",
            formatted_answer="Formatted answer",
            question_prompt="Full question with context",
        ),
        QARecord(
            benchmark_sample_index=2,
            benchmark_sample_hash="hash456",
            subject="Geography",
            system_prompt=None,
            instruction="Identify the country.",
            question="Which country is known as the land of the rising sun?",
            choices=None,
            label="Japan",
            formatted_question="Formatted question without choices",
            formatted_answer="Japan is the answer.",
            question_prompt="Full question without system prompt",
        ),
    ],
)
def test_qa_record_json_serialization(record: QARecord) -> None:
    """Test the JSON serialization & deserialization of QARecord."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "record.json"
        write_as_json(file_path, record.to_dict())
        loaded_record = QARecord.from_dict(read_json_file(file_path))
        assert record == loaded_record


def test_qa_formatter() -> None:
    # pylint: disable=protected-access
    format_cfg = {
        "instructions": {
            "system_prompt": "System prompt",
            "base_inst_template": "Basic instruction template for {{ subject }}",
            "chat_inst_template": "Chat instruction template for {{ subject }}",
        },
        "prompt": {
            "question_template": "Question: {{ question }}",
            "answer_template": "Answer: {{ answer }}",
            "prompt_template": """{{ instruction }}
{%- if examples %}
{%- for example in examples %}

{{ example.question }}
{{ example.answer }}
{%- endfor %}
{%- endif %}

{{ question }}""",
        },
    }

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
                QARecord(
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
            QARecord(
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
            )
        ],
        subject="Geography",
    ) == QARecord(
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
    ) == QARecord(
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
    )


def test_qa_formatter_render_conversation() -> None:
    format_cfg = {
        "instructions": {
            "system_prompt": "System prompt",
            "base_inst_template": "Basic instruction template for {{ subject }}",
            "chat_inst_template": "Chat instruction template for {{ subject }}",
        },
        "prompt": {
            "question_template": "Question: {{ question }}",
            "answer_template": "Answer: {{ answer }}",
            "prompt_template": """{{ instruction }}
{%- if examples %}
{%- for example in examples %}

{{ example.question }}
{{ example.answer }}
{%- endfor %}
{%- endif %}

{{ question }}""",
        },
    }

    base_formatter = QAFormatter(PromptFormatter.BASE, format_cfg=format_cfg)
    assert (
        base_formatter.render_conversation(
            QARecord(
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
        QARecord(
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
