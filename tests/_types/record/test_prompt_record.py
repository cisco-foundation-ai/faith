# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from faith._internal.algo.hash import dict_sha256
from faith._internal.io.json import read_json_file, write_as_json
from faith._types.record.prompt_record import PromptRecord


def test_prompt_record_to_dict() -> None:
    record = PromptRecord(
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
        ancillary_data=None,
    )
    dict_repr = record.to_dict()
    assert dict_repr == {
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
    assert (
        dict_sha256(dict_repr)
        == "735c2c600a48d2e20bfe6d3f43157253b68c69118bb9298dde889fbf6c59b029"
    ), f"Hash of prompt record has changed: {dict_sha256(dict_repr)}"


@pytest.mark.parametrize(
    "record",
    [
        PromptRecord(
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
            ancillary_data={
                "difficulty": "hard",
                "counter": 17,
                "tags": ["philosophy", "metaphor"],
            },
        ),
        PromptRecord(
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
            ancillary_data={},
        ),
        PromptRecord(
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
            ancillary_data=None,
        ),
    ],
)
def test_prompt_record_json_serialization(record: PromptRecord) -> None:
    """Test the JSON serialization & deserialization of PromptRecord."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "record.json"
        write_as_json(file_path, record)
        assert record == PromptRecord.from_dict(read_json_file(file_path))
