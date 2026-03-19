# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Iterable
from typing import Any
from unittest.mock import ANY, patch

import pytest
from datasets import Dataset, DatasetDict, Features, Value

from faith import __version__
from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.benchmark.spec import BenchmarkSpec
from faith._types.config.benchmark import BenchmarkConfig, LAQAConfig, LongAnswerType
from faith._types.config.format import FormatConfig, InstructionsConfig, PromptConfig
from faith._types.config.patterns import (
    AnswerFormat,
    CaptureTransform,
    Disambiguation,
    PatternDef,
)
from faith._types.config.scoring import OutputProcessingConfig, ScoreFnConfig
from faith._types.config.source import HuggingFaceSourceConfig, SourceConfig
from faith._types.model.engine import ModelEngine
from faith._types.model.generation import GenerationMode
from faith._types.model.prompt import PromptFormatter
from faith._types.record.model_response import ChatResponse, GenerationError
from faith._types.record.stats import StatsRecord
from faith.benchmark.categories.long_answer import LABenchmark
from faith.model.base import BaseModel, PromptList
from tests.benchmark.categories.fake_record_maker import make_fake_record


def test_long_answer_benchmark_logits() -> None:
    with pytest.raises(
        AssertionError,
        match="Long answer benchmarks do not support logits/next_token generation mode",
    ):
        LABenchmark(
            spec=BenchmarkSpec(
                name="test-foo",
                generation_mode=GenerationMode.LOGITS,
                prompt_format=PromptFormatter.BASE,
                n_shot=SampleRatio(5),
            ),
            config=BenchmarkConfig(
                laqa_config=LAQAConfig(type=LongAnswerType.FREE_FORM),
                format=FormatConfig(
                    instructions=InstructionsConfig(
                        system_prompt_template="You are a helpful assistant.",
                        base_inst_template="Please respond to the following question.",
                        chat_inst_template="Please respond to the following question in a chat format.",
                    ),
                    prompt=PromptConfig(
                        question_template="Question: {{ question }}",
                        answer_template="Answer: {{ answer }}",
                        prompt_template="{{ instruction }}\n\n{{ question }}",
                    ),
                ),
                output_processing=OutputProcessingConfig(
                    score_fns={
                        "llm_grade": ScoreFnConfig(
                            type="llm_judge",
                            kwargs={
                                "judge_prompt_template": "Compare `{{ correct_answer }}` to `{{ generated_answer }}` on a scale from 1-10.",
                                "judge_model": {
                                    "model_engine": "openai",
                                    "model_path": "gpt-4o",
                                    "engine_kwargs": {"api_num_threads": 1},
                                    "generation_kwargs": {
                                        "temperature": 0.3,
                                        "max_completion_tokens": 1024,
                                    },
                                },
                                "llm_score_range": {"min": 1.0, "max": 10.0},
                                "verdict_formats": [
                                    PatternDef(
                                        pattern=r"(?is)\bSCORE:\s*(\d{1,2})\b",
                                        capture_transform=CaptureTransform(
                                            params=["score"],
                                            expr="{'awarded_points': float(score)}",
                                        ),
                                        disambiguation=Disambiguation.MATCH_LAST,
                                        format_type=AnswerFormat.PROPER,
                                    ),
                                ],
                            },
                        ),
                    },
                ),
            ),
        )


def test_long_answer_benchmark_next_token() -> None:
    with pytest.raises(
        AssertionError,
        match="Long answer benchmarks do not support logits/next_token generation mode",
    ):
        LABenchmark(
            spec=BenchmarkSpec(
                name="test-foo",
                generation_mode=GenerationMode.NEXT_TOKEN,
                prompt_format=PromptFormatter.BASE,
                n_shot=SampleRatio(5),
            ),
            config=BenchmarkConfig(
                laqa_config=LAQAConfig(type=LongAnswerType.FREE_FORM),
                format=FormatConfig(
                    instructions=InstructionsConfig(
                        system_prompt_template="You are a helpful assistant.",
                        base_inst_template="Please respond to the following question.",
                        chat_inst_template="Please respond to the following question in a chat format.",
                    ),
                    prompt=PromptConfig(
                        question_template="Question: {{ question }}",
                        answer_template="Answer: {{ answer }}",
                        prompt_template="{{ instruction }}\n\n{{ question }}",
                    ),
                ),
                output_processing=OutputProcessingConfig(
                    score_fns={
                        "llm_grade": ScoreFnConfig(
                            type="llm_judge",
                            kwargs={
                                "judge_prompt_template": "Compare `{{ correct_answer }}` to `{{ generated_answer }}` on a scale from 1-10.",
                                "judge_model": {
                                    "model_engine": "openai",
                                    "model_path": "gpt-4o",
                                    "engine_kwargs": {"api_num_threads": 1},
                                    "generation_kwargs": {
                                        "temperature": 0.3,
                                        "max_completion_tokens": 1024,
                                    },
                                },
                                "llm_score_range": {"min": 1.0, "max": 10.0},
                                "verdict_formats": [
                                    PatternDef(
                                        pattern=r"(?is)\bSCORE:\s*(\d{1,2})\b",
                                        capture_transform=CaptureTransform(
                                            params=["score"],
                                            expr="{'awarded_points': float(score)}",
                                        ),
                                        disambiguation=Disambiguation.MATCH_LAST,
                                        format_type=AnswerFormat.PROPER,
                                    ),
                                ],
                            },
                        ),
                    },
                ),
            ),
        )


def test_long_answer_benchmark_chat() -> None:
    benchmark = LABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(5),
        ),
        config=BenchmarkConfig(
            laqa_config=LAQAConfig(type=LongAnswerType.FREE_FORM),
            format=FormatConfig(
                instructions=InstructionsConfig(
                    base_inst_template="Please respond to the following question.",
                    chat_inst_template="Please respond to the following question in a chat format.",
                ),
                prompt=PromptConfig(
                    question_template="Question: {{ question }}",
                    answer_template="Answer: {{ answer }}",
                    prompt_template="{{ instruction }}\n\n{{ question }}",
                ),
            ),
            output_processing=OutputProcessingConfig(
                score_fns={
                    "llm_grade": ScoreFnConfig(
                        type="llm_judge",
                        kwargs={
                            "judge_prompt_template": "Compare `{{ correct_answer }}` to `{{ generated_answer }}` on a scale from 1-10.",
                            "judge_model": {
                                "model_engine": "openai",
                                "model_path": "gpt-4o",
                                "engine_kwargs": {"api_num_threads": 1},
                                "generation_kwargs": {
                                    "temperature": 0.3,
                                    "max_completion_tokens": 1024,
                                },
                            },
                            "llm_score_range": {"min": 1.0, "max": 10.0},
                            "verdict_formats": [
                                PatternDef(
                                    pattern=r"(?is)\bSCORE:\s*(\d{1,2})\b",
                                    capture_transform=CaptureTransform(
                                        params=["score"],
                                        expr="{'awarded_points': float(score)}",
                                    ),
                                    disambiguation=Disambiguation.MATCH_LAST,
                                    format_type=AnswerFormat.PROPER,
                                ),
                            ],
                        },
                    ),
                },
            ),
        ),
    )

    assert benchmark.answer_set is None
    assert benchmark.generation_mode == GenerationMode.CHAT_COMP
    assert benchmark.version == __version__


def test_long_answer_benchmark_build_dataset() -> None:
    fake_test_dataset = Dataset.from_dict(
        {
            "question": ["What is the capital of Nepal?", "What is 4+5?"],
            "answer": ["Kathmandu", "9"],
        },
        features=Features({"question": Value("string"), "answer": Value("string")}),
    )
    fake_dev_dataset = Dataset.from_dict(
        {"question": ["What is 'dog' in German?"], "answer": ["Hund"]},
        features=Features({"question": Value("string"), "answer": Value("string")}),
    )
    fake_dataset_dict = DatasetDict(
        {"test": fake_test_dataset, "dev": fake_dev_dataset}
    )

    benchmark_1shot = LABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(1),
        ),
        config=BenchmarkConfig(
            laqa_config=LAQAConfig(type=LongAnswerType.FREE_FORM),
            format=FormatConfig(
                instructions=InstructionsConfig(
                    system_prompt_template="You are a helpful assistant.",
                    base_inst_template="Please respond to the following question.",
                    chat_inst_template="Please respond to the following question in a chat format.",
                ),
                prompt=PromptConfig(
                    question_template="Question: {{ question }}",
                    answer_template="Answer: {{ answer }}",
                    prompt_template="{{ instruction }}\n\n{{ question }}",
                ),
            ),
            source=SourceConfig(
                huggingface=HuggingFaceSourceConfig(
                    path="foo/baz-bar",
                    subset_name="qux",
                    test_split="test",
                    dev_split="dev",
                ),
            ),
            output_processing=OutputProcessingConfig(
                score_fns={
                    "llm_grade": ScoreFnConfig(
                        type="llm_judge",
                        kwargs={
                            "judge_prompt_template": "Compare `{{ correct_answer }}` to `{{ generated_answer }}` on a scale from 1-10.",
                            "judge_model": {
                                "model_engine": "openai",
                                "model_path": "gpt-4o",
                                "engine_kwargs": {"api_num_threads": 1},
                                "generation_kwargs": {
                                    "temperature": 0.3,
                                    "max_completion_tokens": 1024,
                                },
                            },
                            "llm_score_range": {"min": 1.0, "max": 10.0},
                            "verdict_formats": [
                                PatternDef(
                                    pattern=r"(?is)\bSCORE:\s*(\d{1,2})\b",
                                    capture_transform=CaptureTransform(
                                        params=["score"],
                                        expr="{'awarded_points': float(score)}",
                                    ),
                                    disambiguation=Disambiguation.MATCH_LAST,
                                    format_type=AnswerFormat.PROPER,
                                ),
                            ],
                        },
                    ),
                },
            ),
        ),
        seed=42,
    )
    with patch(
        "faith.benchmark.dataset.load.load_dataset",
        return_value=fake_dataset_dict,
    ) as mock_load_dataset:
        dataset_1shot = benchmark_1shot.build_dataset()
        mock_load_dataset.assert_called_once_with("foo/baz-bar", "qux")

        # Compare the questions as dictionaries.
        assert [rec.to_dict() for rec in dataset_1shot.iter_data()] == [
            {
                "benchmark_sample_index": 0,
                "benchmark_sample_hash": ANY,
                "subject": None,
                "system_prompt": "You are a helpful assistant.",
                "instruction": "Please respond to the following question.",
                "question": "What is the capital of Nepal?",
                "choices": None,
                "label": "Kathmandu",
                "formatted_question": "Question: What is the capital of Nepal?",
                "formatted_answer": "Answer: Kathmandu",
                "question_prompt": "Please respond to the following question.\n\nQuestion: What is the capital of Nepal?",
                "ancillary_data": None,
            },
            {
                "benchmark_sample_index": 1,
                "benchmark_sample_hash": ANY,
                "subject": None,
                "system_prompt": "You are a helpful assistant.",
                "instruction": "Please respond to the following question.",
                "question": "What is 4+5?",
                "choices": None,
                "label": "9",
                "formatted_question": "Question: What is 4+5?",
                "formatted_answer": "Answer: 9",
                "question_prompt": "Please respond to the following question.\n\nQuestion: What is 4+5?",
                "ancillary_data": None,
            },
        ]

    benchmark_1shot_no_dev = LABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(1),
        ),
        config=BenchmarkConfig(
            laqa_config=LAQAConfig(type=LongAnswerType.FREE_FORM),
            format=FormatConfig(
                instructions=InstructionsConfig(
                    system_prompt_template="You are a helpful assistant.",
                    base_inst_template="Please respond to the following question.",
                    chat_inst_template="Please respond to the following question in a chat format.",
                ),
                prompt=PromptConfig(
                    question_template="Question: {{ question }}",
                    answer_template="Answer: {{ answer }}",
                    prompt_template="{{ instruction }}\n\n{{ question }}",
                ),
            ),
            source=SourceConfig(
                huggingface=HuggingFaceSourceConfig(
                    path="foo/baz-bar",
                    subset_name="qux",
                    test_split="test",
                ),
            ),
            output_processing=OutputProcessingConfig(
                score_fns={
                    "llm_grade": ScoreFnConfig(
                        type="llm_judge",
                        kwargs={
                            "judge_prompt_template": "Compare `{{ correct_answer }}` to `{{ generated_answer }}` on a scale from 1-10.",
                            "judge_model": {
                                "model_engine": "openai",
                                "model_path": "gpt-4o",
                                "engine_kwargs": {"api_num_threads": 1},
                                "generation_kwargs": {
                                    "temperature": 0.3,
                                    "max_completion_tokens": 1024,
                                },
                            },
                            "llm_score_range": {"min": 1.0, "max": 10.0},
                            "verdict_formats": [
                                PatternDef(
                                    pattern=r"(?is)\bSCORE:\s*(\d{1,2})\b",
                                    capture_transform=CaptureTransform(
                                        params=["score"],
                                        expr="{'awarded_points': float(score)}",
                                    ),
                                    disambiguation=Disambiguation.MATCH_LAST,
                                    format_type=AnswerFormat.PROPER,
                                ),
                            ],
                        },
                    ),
                },
            ),
        ),
        seed=42,
    )
    with patch(
        "faith.benchmark.dataset.load.load_dataset",
        return_value=fake_dataset_dict,
    ) as mock_load_dataset:
        dataset_1shot_no_dev = benchmark_1shot_no_dev.build_dataset()
        mock_load_dataset.assert_called_once_with("foo/baz-bar", "qux")

        # Compare the questions as dictionaries.
        assert [rec.to_dict() for rec in dataset_1shot_no_dev.iter_data()] == [
            {
                "benchmark_sample_index": 0,
                "benchmark_sample_hash": ANY,
                "subject": None,
                "system_prompt": "You are a helpful assistant.",
                "instruction": "Please respond to the following question.",
                "question": "What is the capital of Nepal?",
                "choices": None,
                "label": "Kathmandu",
                "formatted_question": "Question: What is the capital of Nepal?",
                "formatted_answer": "Answer: Kathmandu",
                "question_prompt": "Please respond to the following question.\n\nQuestion: What is the capital of Nepal?",
                "ancillary_data": None,
            },
        ]

    judged_test_dataset = Dataset.from_dict(
        {
            "question": ["What is the capital of Nepal?", "What is 4+5?"],
            "answer": ["Kathmandu", "9"],
        },
        features=Features(
            {
                "question": Value("string"),
                "answer": Value("string"),
            }
        ),
    )
    judged_dev_dataset = Dataset.from_dict(
        {"question": ["What is 'dog' in German?"], "answer": ["Hund"]},
        features=Features({"question": Value("string"), "answer": Value("string")}),
    )
    judged_dataset_dict = DatasetDict(
        {"test": judged_test_dataset, "dev": judged_dev_dataset}
    )
    benchmark_0shot = LABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(0),
        ),
        config=BenchmarkConfig(
            laqa_config=LAQAConfig(type=LongAnswerType.FREE_FORM),
            format=FormatConfig(
                instructions=InstructionsConfig(
                    system_prompt_template="You are a helpful assistant.",
                    base_inst_template="Please respond to the following question.",
                    chat_inst_template="Please respond to the following question in a chat format.",
                ),
                prompt=PromptConfig(
                    question_template="Question: {{ question }}",
                    answer_template="Answer: {{ answer }}",
                    prompt_template="{{ instruction }}\n\n{{ question }}",
                ),
            ),
            source=SourceConfig(
                huggingface=HuggingFaceSourceConfig(
                    path="foo/baz-bar",
                    test_split="test",
                ),
            ),
            output_processing=OutputProcessingConfig(
                score_fns={
                    "llm_grade": ScoreFnConfig(
                        type="llm_judge",
                        kwargs={
                            "judge_prompt_template": "Compare `{{ correct_answer }}` to `{{ generated_answer }}` on a scale from 1-10.",
                            "judge_model": {
                                "model_engine": "openai",
                                "model_path": "gpt-4o",
                                "engine_kwargs": {"api_num_threads": 1},
                                "generation_kwargs": {
                                    "temperature": 0.3,
                                    "max_completion_tokens": 1024,
                                },
                            },
                            "llm_score_range": {"min": 1.0, "max": 10.0},
                            "verdict_formats": [
                                PatternDef(
                                    pattern=r"(?is)\bSCORE:\s*(\d{1,2})\b",
                                    capture_transform=CaptureTransform(
                                        params=["score"],
                                        expr="{'awarded_points': float(score)}",
                                    ),
                                    disambiguation=Disambiguation.MATCH_LAST,
                                    format_type=AnswerFormat.PROPER,
                                ),
                            ],
                        },
                    ),
                },
            ),
        ),
        seed=42,
    )
    with patch(
        "faith.benchmark.dataset.load.load_dataset",
        return_value=judged_dataset_dict,
    ) as mock_load_dataset:
        dataset_0shot = benchmark_0shot.build_dataset(sample_size=1)
        mock_load_dataset.assert_called_once_with("foo/baz-bar", None)

        # Compare the questions as dictionaries.
        assert [rec.to_dict() for rec in dataset_0shot.iter_data()] == [
            {
                "benchmark_sample_index": 1,
                "benchmark_sample_hash": ANY,
                "subject": None,
                "system_prompt": "You are a helpful assistant.",
                "instruction": "Please respond to the following question.",
                "question": "What is 4+5?",
                "choices": None,
                "label": "9",
                "formatted_question": "Question: What is 4+5?",
                "formatted_answer": "Answer: 9",
                "question_prompt": "Please respond to the following question.\n\nQuestion: What is 4+5?",
                "ancillary_data": None,
            },
        ]


class _FakeJudgeModel(BaseModel):
    """A fake model that simulates judge responses for testing purposes."""

    def __init__(self, model_name: str):
        """Initializes the fake judge model."""
        super().__init__(model_name)
        assert model_name == "gpt-4o"

    @property
    def supported_formats(self) -> set[PromptFormatter]:
        """The fake model supports all prompt formats."""
        return set(list(PromptFormatter))

    def query(
        self,
        inputs: PromptList,
        **_kwargs: Any,
    ) -> Iterable[ChatResponse | GenerationError]:
        """Simulates a judge model response for each input prompt."""
        assert len(inputs) == 1, "Expected each query call to have a single input"
        yield ChatResponse(
            output_text="SCORE: 8\n\nSUMMARY: fake response",
            num_output_tokens=25,
            num_prompt_tokens=10,
            num_request_tokens=10,
            response_text="SCORE: 8\n\nSUMMARY: fake response",
            num_response_tokens=25,
            answer_text="SCORE: 8\n\nSUMMARY: fake response",
            num_answer_tokens=25,
            max_token_halt=False,
        )


def test_long_answer_benchmark_process_logs_chat() -> None:
    bench_config = BenchmarkConfig(
        laqa_config=LAQAConfig(type=LongAnswerType.FREE_FORM),
        format=FormatConfig(
            instructions=InstructionsConfig(
                system_prompt_template="You are a helpful assistant.",
                base_inst_template="Please respond to the following question.",
                chat_inst_template="Please respond to the following question in a chat format.",
            ),
            prompt=PromptConfig(
                question_template="Question: {{ question }}",
                answer_template="Answer: {{ answer }}",
                prompt_template="{{ instruction }}\n\n{{ question }}",
            ),
        ),
        output_processing=OutputProcessingConfig(
            score_fns={
                "llm_grade": ScoreFnConfig(
                    type="llm_judge",
                    kwargs={
                        "judge_prompt_template": """You are an expert evaluator tasked with comparing two answers for accuracy and completeness.

GOLDEN ANSWER (Expected/Correct): {{ correct_answer }}

MODEL'S ANSWER (to be compared against the golden answer): {{ generated_answer }}

Evaluate the model's answer against the golden answer and provide a score from 1-10 and a summary of the differences.

Format your response exactly as:
SCORE: [number]
SUMMARY: [your summary text]""",
                        "judge_model": {
                            "model_engine": "openai",
                            "model_path": "gpt-4o",
                            "engine_kwargs": {"api_num_threads": 1},
                            "generation_kwargs": {
                                "temperature": 0.3,
                                "max_completion_tokens": 1024,
                            },
                        },
                        "llm_score_range": {"min": 1.0, "max": 10.0},
                        "verdict_formats": [
                            PatternDef(
                                pattern=r"(?is)\bSCORE:\s*(\d{1,2})\b.*SUMMARY:\s*(.+)",
                                capture_transform=CaptureTransform(
                                    params=["score", "summary"],
                                    expr="{'awarded_points': float(score), 'details': {'summary': summary.strip()}}",
                                ),
                                disambiguation=Disambiguation.MATCH_LAST,
                                format_type=AnswerFormat.PROPER,
                            ),
                        ],
                    },
                ),
            },
        ),
    )

    benchmark_chat = LABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(5),
        ),
        config=bench_config,
    )

    with patch(
        "faith.benchmark.scores.domain_specific.create_model",
        return_value=_FakeJudgeModel("gpt-4o"),
    ) as mock_create_model:
        log_grader = benchmark_chat.log_grader(
            model_format_config=PatternDef(
                pattern=r"(?s)(?:\s*<t>.*</t>(?!.*</t>)|\s*<t>.*)?(.*)",
                disambiguation=Disambiguation.MATCH_ALL,
                format_type=AnswerFormat.PROPER,
            )
        )
        mock_create_model.assert_called_once_with(
            ModelEngine.OPENAI, "gpt-4o", api_num_threads=1
        )

    assert [log.stats for log in [] >> log_grader] == []
    assert [
        log.stats
        for log in [
            make_fake_record(
                data={
                    "label": "foo",
                    "subject": "bar",
                    "question_prompt": "Question: baz",
                },
                model_data={
                    "chat_comp": {
                        "answer_text": "<t>Maybe </t>Answer: bar</t>Answer: foo",
                        "output_text": "<t>Maybe </t>Answer: bar</t>Answer: foo",
                        "num_output_tokens": 3,
                        "max_token_halt": False,
                    }
                },
            ),
            make_fake_record(
                data={
                    "label": "foo",
                    "subject": "bar",
                    "question_prompt": "Question: baz",
                },
                model_data={
                    "chat_comp": {
                        "answer_text": "<t>Maybe Answer: foo",
                        "output_text": "<t>Maybe Answer: foo",
                        "num_output_tokens": 3,
                        "max_token_halt": True,
                    }
                },
            ),
            make_fake_record(
                data={"label": "bar", "question_prompt": "Question: baz"},
                model_data={
                    "chat_comp": {
                        "answer_text": "<answer>BaZ</answer>",
                        "output_text": "<answer>BaZ</answer>",
                        "num_output_tokens": 5,
                        "max_token_halt": False,
                    }
                },
            ),
        ]
        >> log_grader
    ] == [
        StatsRecord(
            label="foo",
            prediction="Answer: foo",
            answer_format=AnswerFormat.PROPER,
            subject="bar",
            num_output_tokens=3,
            max_token_halt=False,
            scores={
                "llm_grade": {
                    "value": pytest.approx(7 / 9),
                    "raw_value": 8.0,
                    "summary_details": {"summary": "fake response"},
                    "full_response": "SCORE: 8\n\nSUMMARY: fake response",
                },
            },
        ),
        StatsRecord(
            label="foo",
            prediction="",
            answer_format=AnswerFormat.PROPER,
            subject="bar",
            num_output_tokens=3,
            max_token_halt=True,
            scores={
                "llm_grade": {
                    "value": pytest.approx(7 / 9),
                    "raw_value": 8.0,
                    "summary_details": {"summary": "fake response"},
                    "full_response": "SCORE: 8\n\nSUMMARY: fake response",
                },
            },
        ),
        StatsRecord(
            label="bar",
            prediction="<answer>BaZ</answer>",
            answer_format=AnswerFormat.PROPER,
            num_output_tokens=5,
            max_token_halt=False,
            scores={
                "llm_grade": {
                    "value": pytest.approx(7 / 9),
                    "raw_value": 8.0,
                    "summary_details": {"summary": "fake response"},
                    "full_response": "SCORE: 8\n\nSUMMARY: fake response",
                },
            },
        ),
    ]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_long_answer_benchmark_grade_aggregator() -> None:
    bench_config = BenchmarkConfig(
        laqa_config=LAQAConfig(type=LongAnswerType.FREE_FORM),
        format=FormatConfig(
            instructions=InstructionsConfig(
                system_prompt_template="You are a helpful assistant.",
                base_inst_template="Please respond to the following question.",
                chat_inst_template="Please respond to the following question in a chat format.",
            ),
            prompt=PromptConfig(
                question_template="Question: {{ question }}",
                answer_template="Answer: {{ answer }}",
                prompt_template="{{ instruction }}\n\n{{ question }}",
            ),
        ),
        output_processing=OutputProcessingConfig(
            score_fns={
                "llm_grade": ScoreFnConfig(
                    type="llm_judge",
                    kwargs={
                        "judge_prompt_template": """You are an expert evaluator tasked with comparing two answers for accuracy and completeness.

GOLDEN ANSWER (Expected/Correct): {{ correct_answer }}

MODEL'S ANSWER (to be compared against the golden answer): {{ generated_answer }}

Evaluate the model's answer against the golden answer and provide a score from 1-10 and a summary of the differences.

Format your response exactly as:
SCORE: [number]
SUMMARY: [your summary text]""",
                        "judge_model": {
                            "model_engine": "openai",
                            "model_path": "gpt-4o",
                            "engine_kwargs": {"api_num_threads": 1},
                            "generation_kwargs": {
                                "temperature": 0.3,
                                "max_completion_tokens": 1024,
                            },
                        },
                        "llm_score_range": {"min": 1.0, "max": 10.0},
                        "verdict_formats": [
                            PatternDef(
                                pattern=r"(?is)\bSCORE:\s*(\d{1,2})\b.*SUMMARY:\s*(.+)",
                                capture_transform=CaptureTransform(
                                    params=["score", "summary"],
                                    expr="{'awarded_points': float(score), 'details': {'summary': summary.strip()}}",
                                ),
                                disambiguation=Disambiguation.MATCH_LAST,
                                format_type=AnswerFormat.PROPER,
                            ),
                        ],
                    },
                ),
            },
        ),
    )

    benchmark_chat = LABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(5),
        ),
        config=bench_config,
    )

    with patch(
        "faith.benchmark.scores.domain_specific.create_model",
        return_value=_FakeJudgeModel("gpt-4o"),
    ) as mock_create_model:
        aggregator = benchmark_chat.grade_aggregator()
        mock_create_model.assert_called_once_with(
            ModelEngine.OPENAI, "gpt-4o", api_num_threads=1
        )

    assert [] >> aggregator == {
        "format_count": {
            "improper": 0,
            "inferred": 0,
            "invalid": 0,
            "proper": 0,
        },
        "format_rate": {
            "improper": pytest.approx(float("nan"), nan_ok=True),
            "inferred": pytest.approx(float("nan"), nan_ok=True),
            "invalid": pytest.approx(float("nan"), nan_ok=True),
            "proper": pytest.approx(float("nan"), nan_ok=True),
        },
        "mean_llm_grade": pytest.approx(float("nan"), nan_ok=True),
        "median_llm_grade": pytest.approx(float("nan"), nan_ok=True),
        "stddev_llm_grade": pytest.approx(float("nan"), nan_ok=True),
        "query_count": 0,
    }

    assert [
        StatsRecord(
            label="foo bar",
            prediction="foo bar baz",
            answer_format=AnswerFormat.PROPER,
            num_output_tokens=4,
            max_token_halt=False,
            scores={
                "llm_grade": {
                    "value": 0.8,
                    "raw_value": 8.0,
                    "min_value": 0.0,
                    "max_value": 10.0,
                },
            },
        ),
        StatsRecord(
            label="a b c d",
            prediction="a b c d e",
            answer_format=AnswerFormat.PROPER,
            num_output_tokens=5,
            max_token_halt=False,
            scores={
                "llm_grade": {
                    "value": 1.0,
                    "raw_value": 5.0,
                    "min_value": 1.0,
                    "max_value": 5.0,
                },
            },
        ),
        StatsRecord(
            label="one two three",
            prediction="ooops",
            answer_format=AnswerFormat.PROPER,
            num_output_tokens=6,
            max_token_halt=False,
            scores={
                "llm_grade": {
                    "value": 0.1,
                    "raw_value": 2.0,
                    "min_value": 1.0,
                    "max_value": 11.0,
                },
            },
        ),
    ] >> aggregator == {
        "format_count": {
            "improper": 0,
            "inferred": 0,
            "invalid": 0,
            "proper": 3,
        },
        "format_rate": {
            "improper": pytest.approx(0),
            "inferred": pytest.approx(0),
            "invalid": pytest.approx(0),
            "proper": pytest.approx(1),
        },
        "mean_output_tokens": pytest.approx(5),
        "mean_llm_grade": pytest.approx(19 / 30),
        "median_llm_grade": pytest.approx(4 / 5),
        "stddev_llm_grade": pytest.approx(math.sqrt(67 / 2) / 15),
        "query_count": 3,
        "rate_max_token_halt": pytest.approx(0),
    }
