# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, Iterable, cast
from unittest.mock import ANY, patch

import pytest
from datasets import Dataset, DatasetDict, Features, Value

from faith import __version__
from faith._internal.algo.matching import AnswerFormat
from faith._internal.types.flags import GenerationMode, SampleRatio
from faith.benchmark.benchmark import BenchmarkSpec
from faith.benchmark.categories.long_answer import LABenchmark
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import BaseModel, ChatResponse, GenerationError, PromptList


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
            config={
                "laqa_config": {"type": "free_form"},
                "format": {
                    "instructions": {
                        "system_prompt": "You are a helpful assistant.",
                        "base_inst_template": "Please answer the following question.",
                        "chat_inst_template": "Please answer the following question in a chat format.",
                    },
                    "prompt": {
                        "question_template": "Question: {{ question }}",
                        "answer_template": "Answer: {{ answer }}",
                        "prompt_template": "{{ instruction }}\n\n{{ question }}",
                    },
                },
            },
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
            config={
                "laqa_config": {"type": "free_form"},
                "format": {
                    "instructions": {
                        "system_prompt": "You are a helpful assistant.",
                        "base_inst_template": "Please answer the following question.",
                        "chat_inst_template": "Please answer the following question in a chat format.",
                    },
                    "prompt": {
                        "question_template": "Question: {{ question }}",
                        "answer_template": "Answer: {{ answer }}",
                        "prompt_template": "{{ instruction }}\n\n{{ question }}",
                    },
                },
            },
        )


def test_long_answer_benchmark_chat() -> None:
    benchmark = LABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(5),
        ),
        config={
            "laqa_config": {"type": "free_form"},
            "format": {
                "instructions": {
                    "base_inst_template": "Please answer the following question.",
                    "chat_inst_template": "Please answer the following question in a chat format.",
                },
                "prompt": {
                    "question_template": "Question: {{ question }}",
                    "answer_template": "Answer: {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
        },
    )

    assert benchmark.answer_set is None
    assert benchmark.generation_mode == GenerationMode.CHAT_COMPLETION
    assert benchmark.version == __version__


def test_long_answer_benchmark_build_dataset() -> None:
    fake_test_dataset = Dataset.from_dict(
        {
            "question": ["What is the capital of Austria?", "What is 1+2?"],
            "answer": ["Vienna", "3"],
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
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(1),
        ),
        config={
            "laqa_config": {"type": "free_form"},
            "format": {
                "instructions": {
                    "system_prompt": "You are a helpful assistant.",
                    "base_inst_template": "Please answer the following question.",
                    "chat_inst_template": "Please answer the following question in a chat format.",
                },
                "prompt": {
                    "question_template": "Question: {{ question }}",
                    "answer_template": "Answer: {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
            "source": {
                "huggingface": {
                    "path": "foo/bar-baz",
                    "subset_name": "qux",
                    "test_split": "test",
                    "dev_split": "dev",
                },
            },
        },
        seed=42,
    )
    with patch(
        "faith.benchmark.dataset.load.load_dataset",
        return_value=fake_dataset_dict,
    ) as mock_load_dataset:
        dataset_1shot = benchmark_1shot.build_dataset()
        mock_load_dataset.assert_called_once_with("foo/bar-baz", "qux")

        # Compare the questions as dictionaries.
        assert [q.to_dict() for q in dataset_1shot.iter_data()] == [
            {
                "benchmark_sample_index": 0,
                "benchmark_sample_hash": ANY,
                "subject": None,
                "system_prompt": "You are a helpful assistant.",
                "instruction": "Please answer the following question.",
                "question": "What is the capital of Austria?",
                "choices": None,
                "label": "Vienna",
                "formatted_question": "Question: What is the capital of Austria?",
                "formatted_answer": "Answer: Vienna",
                "question_prompt": "Please answer the following question.\n\nQuestion: What is the capital of Austria?",
                "judge_prompt_template": None,
                "max_points": None,
            },
            {
                "benchmark_sample_index": 1,
                "benchmark_sample_hash": ANY,
                "subject": None,
                "system_prompt": "You are a helpful assistant.",
                "instruction": "Please answer the following question.",
                "question": "What is 1+2?",
                "choices": None,
                "label": "3",
                "formatted_question": "Question: What is 1+2?",
                "formatted_answer": "Answer: 3",
                "question_prompt": "Please answer the following question.\n\nQuestion: What is 1+2?",
                "judge_prompt_template": None,
                "max_points": None,
            },
        ]

    benchmark_1shot_no_dev = LABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(1),
        ),
        config={
            "laqa_config": {"type": "free_form"},
            "format": {
                "instructions": {
                    "system_prompt": "You are a helpful assistant.",
                    "base_inst_template": "Please answer the following question.",
                    "chat_inst_template": "Please answer the following question in a chat format.",
                },
                "prompt": {
                    "question_template": "Question: {{ question }}",
                    "answer_template": "Answer: {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
            "source": {
                "huggingface": {
                    "path": "foo/bar-baz",
                    "subset_name": "qux",
                    "test_split": "test",
                },
            },
        },
        seed=42,
    )
    with patch(
        "faith.benchmark.dataset.load.load_dataset",
        return_value=fake_dataset_dict,
    ) as mock_load_dataset:
        dataset_1shot_no_dev = benchmark_1shot_no_dev.build_dataset()
        mock_load_dataset.assert_called_once_with("foo/bar-baz", "qux")

        # Compare the questions as dictionaries.
        assert [q.to_dict() for q in dataset_1shot_no_dev.iter_data()] == [
            {
                "benchmark_sample_index": 0,
                "benchmark_sample_hash": ANY,
                "subject": None,
                "system_prompt": "You are a helpful assistant.",
                "instruction": "Please answer the following question.",
                "question": "What is the capital of Austria?",
                "choices": None,
                "label": "Vienna",
                "formatted_question": "Question: What is the capital of Austria?",
                "formatted_answer": "Answer: Vienna",
                "question_prompt": "Please answer the following question.\n\nQuestion: What is the capital of Austria?",
                "judge_prompt_template": None,
                "max_points": None,
            },
        ]

    judged_test_dataset = Dataset.from_dict(
        {
            "question": ["What is the capital of Austria?", "What is 1+2?"],
            "answer": ["Vienna", "3"],
            "judge_prompt_template": [
                "Score '{{ generated_answer }}' compared to correct answer '{{ correct_answer }}' on a scale of 0-1?",
                "Score '{{ generated_answer }}' compared to correct answer '{{ correct_answer }}' on a scale of 0-1?",
            ],
            "max_points": [1, 1],
        },
        features=Features(
            {
                "question": Value("string"),
                "answer": Value("string"),
                "judge_prompt_template": Value("string"),
                "max_points": Value("float64"),
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
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(0),
        ),
        config={
            "laqa_config": {"type": "free_form"},
            "format": {
                "instructions": {
                    "system_prompt": "You are a helpful assistant.",
                    "base_inst_template": "Please answer the following question.",
                    "chat_inst_template": "Please answer the following question in a chat format.",
                },
                "prompt": {
                    "question_template": "Question: {{ question }}",
                    "answer_template": "Answer: {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
            "source": {
                "huggingface": {
                    "path": "foo/bar-baz",
                    "test_split": "test",
                },
            },
        },
        seed=42,
    )
    with patch(
        "faith.benchmark.dataset.load.load_dataset",
        return_value=judged_dataset_dict,
    ) as mock_load_dataset:
        dataset_0shot = benchmark_0shot.build_dataset(sample_size=1)
        mock_load_dataset.assert_called_once_with("foo/bar-baz", None)

        # Compare the questions as dictionaries.
        assert [q.to_dict() for q in dataset_0shot.iter_data()] == [
            {
                "benchmark_sample_index": 1,
                "benchmark_sample_hash": ANY,
                "subject": None,
                "system_prompt": "You are a helpful assistant.",
                "instruction": "Please answer the following question.",
                "question": "What is 1+2?",
                "choices": None,
                "label": "3",
                "formatted_question": "Question: What is 1+2?",
                "formatted_answer": "Answer: 3",
                "question_prompt": "Please answer the following question.\n\nQuestion: What is 1+2?",
                "judge_prompt_template": "Score '{{ generated_answer }}' compared to correct answer '{{ correct_answer }}' on a scale of 0-1?",
                "max_points": 1.0,
            },
        ]


class _FakeJudgeModel(BaseModel):
    """A fake model that simulates judge responses for testing purposes."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert model_name == "gpt-4o"

    @property
    def supported_formats(self) -> set[PromptFormatter]:
        return set(list(PromptFormatter))

    def query(
        self,
        inputs: PromptList,
        **kwargs: Any,
    ) -> Iterable[ChatResponse | GenerationError]:
        assert len(inputs) == 1, "Expected each query call to have a single input"
        yield ChatResponse(
            prompt_token_ids=None,
            num_prompt_tokens=10,
            prompt_text=None,
            output_token_ids=None,
            num_output_tokens=25,
            output_text="SCORE: 8\n\nSUMMARY: fake response",
            request_token_ids=None,
            num_request_tokens=10,
            request_text=None,
            response_token_ids=None,
            num_response_tokens=25,
            response_text="SCORE: 8\n\nSUMMARY: fake response",
            answer_token_ids=None,
            num_answer_tokens=25,
            answer_text="SCORE: 8\n\nSUMMARY: fake response",
            max_token_halt=False,
        )


def test_long_answer_benchmark_process_logs_chat() -> None:
    bench_config = {
        "laqa_config": {"type": "free_form"},
        "format": {
            "instructions": {
                "system_prompt": "You are a helpful assistant.",
                "base_inst_template": "Please answer the following question.",
                "chat_inst_template": "Please answer the following question in a chat format.",
            },
            "prompt": {
                "question_template": "Question: {{ question }}",
                "answer_template": "Answer: {{ answer }}",
                "prompt_template": "{{ instruction }}\n\n{{ question }}",
            },
        },
        "output_processing": {
            "answer_judge": {
                "default_prompt_template": """You are an expert evaluator tasked with comparing two answers for accuracy and completeness.

GOLDEN ANSWER (Expected/Correct): {{ correct_answer }}

MODEL'S ANSWER (to be compared against the golden answer): {{ generated_answer }}

Evaluate the model's answer against the golden answer and provide a score from 1-10 and a summary of the differences.

Format your response exactly as:
SCORE: [number]
SUMMARY: [your summary text]""",
                "default_max_score": 10.0,
                "judge_model": {
                    "model_engine": "openai",
                    "model_path": "gpt-4o",
                    "engine_kwargs": {"api_num_threads": 1},
                    "generation_kwargs": {
                        "temperature": 0.3,
                        "max_completion_tokens": 1024,
                    },
                },
                "verdict_formats": [
                    {
                        "pattern": r"(?is)\bSCORE:\s*(\d{1,2})\b.*SUMMARY:\s*(.+)",
                        "capture_transform": {
                            "params": ["score", "summary"],
                            "expr": "float(score), summary.strip()",
                        },
                        "match_disambiguation": "match_last",
                        "format_type": "proper",
                    },
                ],
            },
        },
    }

    benchmark_chat = LABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(5),
        ),
        config=bench_config,
    )

    with patch(
        "faith.model.model_engine.ModelEngine.OPENAI.create_model",
        return_value=_FakeJudgeModel("gpt-4o"),
    ) as mock_create_model:
        log_grader = benchmark_chat.log_grader(
            model_format_config={
                "pattern": r"(?s)(?:\s*<t>.*</t>(?!.*</t>)|\s*<t>.*)?(.*)",
                "match_disambiguation": "match_all",
                "format_type": "proper",
            }
        )
        mock_create_model.assert_called_once_with("gpt-4o", api_num_threads=1)

    assert [log["stats"] for log in [] >> log_grader] == []
    assert [
        log["stats"]
        for log in cast(
            list[dict],
            [
                {
                    "data": {
                        "label": "foo",
                        "subject": "bar",
                        "question_prompt": "Question: baz",
                    },
                    "model_data": {
                        "chat_comp": {
                            "output_text": "<t>Maybe </t>Answer: bar</t>Answer: foo",
                            "num_output_tokens": 3,
                            "max_token_halt": False,
                        }
                    },
                },
                {
                    "data": {
                        "label": "foo",
                        "subject": "bar",
                        "question_prompt": "Question: baz",
                    },
                    "model_data": {
                        "chat_comp": {
                            "output_text": "<t>Maybe Answer: foo",
                            "num_output_tokens": 3,
                            "max_token_halt": True,
                        }
                    },
                },
                {
                    "data": {"label": "bar", "question_prompt": "Question: baz"},
                    "model_data": {
                        "chat_comp": {
                            "output_text": "<answer>BaZ</answer>",
                            "num_output_tokens": 5,
                            "max_token_halt": False,
                        }
                    },
                },
            ],
        )
        >> log_grader
    ] == [
        {
            "answer_format": AnswerFormat.PROPER,
            "awarded_points": 8.0,
            "comments": "fake response",
            "label": "foo",
            "max_points": 10.0,
            "max_token_halt": False,
            "num_output_tokens": 3,
            "prediction": "Answer: foo",
            "subject": "bar",
        },
        {
            "answer_format": AnswerFormat.PROPER,
            "awarded_points": 8.0,
            "comments": "fake response",
            "label": "foo",
            "max_points": 10.0,
            "max_token_halt": True,
            "num_output_tokens": 3,
            "prediction": "",
            "subject": "bar",
        },
        {
            "answer_format": AnswerFormat.PROPER,
            "awarded_points": 8.0,
            "comments": "fake response",
            "label": "bar",
            "max_points": 10.0,
            "max_token_halt": False,
            "num_output_tokens": 5,
            "prediction": "<answer>BaZ</answer>",
            "subject": None,
        },
    ]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_long_answer_benchmark_grade_aggregator() -> None:
    bench_config = {
        "laqa_config": {"type": "free_form"},
        "format": {
            "instructions": {
                "system_prompt": "You are a helpful assistant.",
                "base_inst_template": "Please answer the following question.",
                "chat_inst_template": "Please answer the following question in a chat format.",
            },
            "prompt": {
                "question_template": "Question: {{ question }}",
                "answer_template": "Answer: {{ answer }}",
                "prompt_template": "{{ instruction }}\n\n{{ question }}",
            },
        },
        "output_processing": {
            "answer_judge": {
                "default_prompt_template": """You are an expert evaluator tasked with comparing two answers for accuracy and completeness.

GOLDEN ANSWER (Expected/Correct): {{ correct_answer }}

MODEL'S ANSWER (to be compared against the golden answer): {{ generated_answer }}

Evaluate the model's answer against the golden answer and provide a score from 1-10 and a summary of the differences.

Format your response exactly as:
SCORE: [number]
SUMMARY: [your summary text]""",
                "default_max_score": 10.0,
                "judge_model": {
                    "model_engine": "openai",
                    "model_path": "gpt-4o",
                    "engine_kwargs": {"api_num_threads": 1},
                    "generation_kwargs": {
                        "temperature": 0.3,
                        "max_completion_tokens": 1024,
                    },
                },
                "verdict_formats": [
                    {
                        "pattern": r"(?is)\bSCORE:\s*(\d{1,2})\b.*SUMMARY:\s*(.+)",
                        "capture_transform": {
                            "params": ["score", "summary"],
                            "expr": "float(score), summary.strip()",
                        },
                        "match_disambiguation": "match_last",
                        "format_type": "proper",
                    },
                ],
            },
        },
    }

    benchmark_chat = LABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(5),
        ),
        config=bench_config,
    )
    metric_aggregator = benchmark_chat.grade_aggregator()

    assert [] >> metric_aggregator == {
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
        "per_question_grade_stats": {
            "mean": pytest.approx(float("nan"), nan_ok=True),
            "median": pytest.approx(float("nan"), nan_ok=True),
            "std": pytest.approx(float("nan"), nan_ok=True),
        },
        "query_count": 0,
        "weighted_grade": pytest.approx(float("nan"), nan_ok=True),
    }

    assert [
        {
            "stats": {
                "awarded_points": 8.0,
                "comments": "fake response",
                "label": "foo bar",
                "max_points": 10.0,
                "max_token_halt": False,
                "num_output_tokens": 4,
                "prediction": "foo bar baz",
                "answer_format": AnswerFormat.PROPER,
            }
        },
        {
            "stats": {
                "awarded_points": 5.0,
                "comments": "fake response",
                "label": "a b c d",
                "max_points": 5.0,
                "max_token_halt": False,
                "num_output_tokens": 5,
                "prediction": "a b c d e",
                "answer_format": AnswerFormat.PROPER,
            }
        },
        {
            "stats": {
                "awarded_points": 1.0,
                "comments": "fake response",
                "label": "one two three",
                "max_points": 10.0,
                "max_token_halt": False,
                "num_output_tokens": 6,
                "prediction": "ooops",
                "answer_format": AnswerFormat.PROPER,
            }
        },
    ] >> metric_aggregator == {
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
        "per_question_grade_stats": {
            "mean": pytest.approx(19 / 30),
            "median": pytest.approx(4 / 5),
            "std": pytest.approx(math.sqrt(67 / 2) / 15),
        },
        "query_count": 3,
        "rate_max_token_halt": pytest.approx(0),
        "weighted_grade": pytest.approx(14 / 25),
    }
