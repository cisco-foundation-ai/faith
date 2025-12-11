# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import cast
from unittest.mock import ANY, patch

import pytest
from datasets import Dataset, DatasetDict, Features, Value

from faith import __version__
from faith._internal.algo.matching import AnswerFormat
from faith._internal.types.flags import GenerationMode, SampleRatio
from faith.benchmark.benchmark import BenchmarkSpec
from faith.benchmark.categories.short_answer import SABenchmark
from faith.benchmark.formatting.prompt import PromptFormatter


def test_short_answer_benchmark_logits() -> None:
    with pytest.raises(
        AssertionError,
        match="Short answer benchmarks do not support logits/next_token generation mode",
    ):
        SABenchmark(
            spec=BenchmarkSpec(
                name="test-foo",
                generation_mode=GenerationMode.LOGITS,
                prompt_format=PromptFormatter.BASE,
                n_shot=SampleRatio(5),
            ),
            config={
                "saqa_config": {"type": "string_match"},
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


def test_short_answer_benchmark_next_token() -> None:
    with pytest.raises(
        AssertionError,
        match="Short answer benchmarks do not support logits/next_token generation mode",
    ):
        SABenchmark(
            spec=BenchmarkSpec(
                name="test-foo",
                generation_mode=GenerationMode.NEXT_TOKEN,
                prompt_format=PromptFormatter.BASE,
                n_shot=SampleRatio(5),
            ),
            config={
                "saqa_config": {"type": "string_match"},
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


def test_short_answer_benchmark_chat() -> None:
    benchmark = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(5),
        ),
        config={
            "saqa_config": {"type": "string_match"},
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


def test_short_answer_benchmark_build_dataset() -> None:
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

    benchmark_1shot = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(1),
        ),
        config={
            "saqa_config": {"type": "string_match"},
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
                "ancillary_data": None,
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
                "ancillary_data": None,
            },
        ]

    benchmark_1shot_no_dev = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(1),
        ),
        config={
            "saqa_config": {"type": "string_match"},
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
                "ancillary_data": None,
            },
        ]

    benchmark_0shot = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(0),
        ),
        config={
            "saqa_config": {"type": "string_match"},
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
        return_value=fake_dataset_dict,
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
                "ancillary_data": None,
            },
        ]


def test_short_answer_benchmark_process_logs_chat() -> None:
    bench_config = {
        "saqa_config": {"type": "string_match"},
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
            "answer_formats": [
                {
                    "pattern": r"Answer:\s*(\w+)\b",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().lower()"},
                    "match_disambiguation": "match_first",
                    "format_type": "proper",
                },
                {
                    "pattern": r"<answer>(\w+)</answer>",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().lower()"},
                    "match_disambiguation": "match_last",
                    "format_type": "improper",
                },
            ],
        },
    }

    benchmark_chat = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(5),
        ),
        config=bench_config,
    )
    log_grader = benchmark_chat.log_grader(
        model_format_config={
            "pattern": r"(?s)(?:\s*<t>.*</t>(?!.*</t>)|\s*<t>.*)?(.*)",
            "match_disambiguation": "match_all",
            "format_type": "proper",
        }
    )

    assert [log["stats"] for log in [] >> log_grader] == []
    assert [
        log["stats"]
        for log in cast(
            list[dict],
            [
                {
                    "data": {"label": "foo", "subject": "bar"},
                    "model_data": {
                        "chat_comp": {
                            "output_text": "<t>Maybe </t>Answer: bar</t>Answer: foo",
                            "num_output_tokens": 3,
                            "max_token_halt": False,
                        }
                    },
                },
                {
                    "data": {"label": "foo", "subject": "bar"},
                    "model_data": {
                        "chat_comp": {
                            "output_text": "<t>Maybe Answer: foo",
                            "num_output_tokens": 3,
                            "max_token_halt": True,
                        }
                    },
                },
                {
                    "data": {"label": "bar"},
                    "model_data": {
                        "chat_comp": {
                            "output_text": "<answer>BaZ</answer>",
                            "num_output_tokens": 5,
                            "max_token_halt": False,
                        }
                    },
                },
                {
                    "data": {"label": "baz"},
                    "model_data": {
                        "chat_comp": {
                            "output_text": "uhm... I have no earthly idea",
                            "num_output_tokens": 8,
                            "max_token_halt": True,
                        },
                    },
                },
                {
                    "data": {"label": "qux"},
                    "model_data": {"error": {"title": "No response from model"}},
                },
                {
                    "stats": {
                        "answer_format": "proper",
                        "label": "aaa",
                        "prediction": "b",
                        "subject": "esperanto",
                    },
                },
            ],
        )
        >> log_grader
    ] == [
        {
            "answer_format": AnswerFormat.PROPER,
            "label": "foo",
            "max_token_halt": False,
            "num_output_tokens": 3,
            "prediction": "foo",
            "subject": "bar",
        },
        {
            "answer_format": AnswerFormat.INVALID,
            "label": "foo",
            "max_token_halt": True,
            "num_output_tokens": 3,
            "prediction": None,
            "subject": "bar",
        },
        {
            "answer_format": AnswerFormat.IMPROPER,
            "label": "bar",
            "max_token_halt": False,
            "num_output_tokens": 5,
            "prediction": "baz",
            "subject": None,
        },
        {
            "answer_format": AnswerFormat.INVALID,
            "label": "baz",
            "max_token_halt": True,
            "num_output_tokens": 8,
            "prediction": None,
            "subject": None,
        },
        {
            "answer_format": AnswerFormat.INVALID,
            "label": "qux",
            "max_token_halt": False,
            "num_output_tokens": 0,
            "prediction": None,
            "subject": None,
        },
        {
            "answer_format": AnswerFormat.PROPER,
            "label": "aaa",
            "prediction": "b",
            "subject": "esperanto",
        },
    ]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_short_answer_benchmark_grade_aggregator_string_match() -> None:
    bench_config = {
        "saqa_config": {"type": "string_match"},
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
            "answer_formats": [
                {
                    "pattern": r"Answer:\s*(\w+)\b",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_first",
                    "format_type": "proper",
                },
                {
                    "pattern": r"<answer>(\w+)</answer>",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_last",
                    "format_type": "improper",
                },
            ],
        },
    }

    benchmark_chat = SABenchmark(
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
        "accuracy": pytest.approx(float("nan"), nan_ok=True),
        "format_breakdown_count": {
            "improper": {
                "correct": 0,
                "incorrect": 0,
            },
            "inferred": {
                "correct": 0,
                "incorrect": 0,
            },
            "invalid": {
                "correct": 0,
                "incorrect": 0,
            },
            "proper": {
                "correct": 0,
                "incorrect": 0,
            },
        },
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
        "lenient_accuracy": pytest.approx(float("nan"), nan_ok=True),
        "query_count": 0,
    }

    assert [
        {
            "stats": {
                "label": "A",
                "max_token_halt": True,
                "num_output_tokens": 10,
                "prediction": "A",
                "answer_format": AnswerFormat.PROPER,
                "subject": "esperanto",
            }
        },
        {
            "stats": {
                "label": "B",
                "max_token_halt": True,
                "num_output_tokens": 10,
                "prediction": "C",
                "answer_format": AnswerFormat.PROPER,
                "subject": "esperanto",
            }
        },
        {
            "stats": {
                "label": "B",
                "max_token_halt": False,
                "num_output_tokens": 1,
                "prediction": "B",
                "answer_format": AnswerFormat.IMPROPER,
                "subject": "esperanto",
            }
        },
    ] >> metric_aggregator == {
        "accuracy": pytest.approx(1 / 3),
        "format_breakdown_count": {
            "improper": {
                "correct": 1,
                "incorrect": 0,
            },
            "inferred": {
                "correct": 0,
                "incorrect": 0,
            },
            "invalid": {
                "correct": 0,
                "incorrect": 0,
            },
            "proper": {
                "correct": 1,
                "incorrect": 1,
            },
        },
        "format_count": {
            "improper": 1,
            "inferred": 0,
            "invalid": 0,
            "proper": 2,
        },
        "format_rate": {
            "improper": pytest.approx(1 / 3),
            "inferred": pytest.approx(0),
            "invalid": pytest.approx(0),
            "proper": pytest.approx(2 / 3),
        },
        "lenient_accuracy": pytest.approx(2 / 3),
        "mean_output_tokens": pytest.approx(7),
        "query_count": 3,
        "rate_max_token_halt": pytest.approx(2 / 3),
    }


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_short_answer_benchmark_grade_aggregator_label_set() -> None:
    bench_config = {
        "saqa_config": {"type": "label_set"},
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
            "answer_formats": [
                {
                    "pattern": r"Answer:\s*(\w+)\b",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_first",
                    "format_type": "proper",
                },
                {
                    "pattern": r"<answer>(\w+)</answer>",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_last",
                    "format_type": "improper",
                },
            ],
            "score_fns": {
                "jaccard_index": {"type": "jaccard"},
            },
        },
    }

    benchmark_chat = SABenchmark(
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
        "accuracy": pytest.approx(float("nan"), nan_ok=True),
        "format_breakdown_count": {
            "improper": {
                "correct": 0,
                "incorrect": 0,
            },
            "inferred": {
                "correct": 0,
                "incorrect": 0,
            },
            "invalid": {
                "correct": 0,
                "incorrect": 0,
            },
            "proper": {
                "correct": 0,
                "incorrect": 0,
            },
        },
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
        "mean_jaccard_index": pytest.approx(float("nan"), nan_ok=True),
        "median_jaccard_index": pytest.approx(float("nan"), nan_ok=True),
        "micro_f1": pytest.approx(float("nan"), nan_ok=True),
        "query_count": 0,
    }

    assert [
        {
            "stats": {
                "label": ["A", "B"],
                "max_token_halt": False,
                "num_output_tokens": 13,
                "prediction": ["B", "C"],
                "answer_format": AnswerFormat.PROPER,
                "scores": {"jaccard_index": {"value": 1 / 3}},
            }
        },
        {
            "stats": {
                "label": ["B"],
                "max_token_halt": True,
                "num_output_tokens": 17,
                "prediction": ["B"],
                "answer_format": AnswerFormat.PROPER,
                "scores": {"jaccard_index": {"value": 1 / 2}},
            }
        },
        {
            "stats": {
                "label": ["B", "C"],
                "max_token_halt": False,
                "num_output_tokens": 9,
                "prediction": ["A"],
                "answer_format": AnswerFormat.IMPROPER,
                "scores": {"jaccard_index": {"value": 0}},
            }
        },
    ] >> metric_aggregator == {
        "accuracy": pytest.approx(1 / 3),
        "format_breakdown_count": {
            "improper": {
                "correct": 0,
                "incorrect": 1,
            },
            "inferred": {
                "correct": 0,
                "incorrect": 0,
            },
            "invalid": {
                "correct": 0,
                "incorrect": 0,
            },
            "proper": {
                "correct": 1,
                "incorrect": 1,
            },
        },
        "format_count": {
            "improper": 1,
            "inferred": 0,
            "invalid": 0,
            "proper": 2,
        },
        "format_rate": {
            "improper": pytest.approx(1 / 3),
            "inferred": pytest.approx(0),
            "invalid": pytest.approx(0),
            "proper": pytest.approx(2 / 3),
        },
        "mean_jaccard_index": pytest.approx(5 / 18),
        "mean_output_tokens": pytest.approx(13),
        "median_jaccard_index": pytest.approx(1 / 3),
        "micro_f1": pytest.approx(4 / 9),
        "query_count": 3,
        "rate_max_token_halt": pytest.approx(1 / 3),
    }


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_short_answer_benchmark_grade_aggregator_domain_specific() -> None:
    bench_config = {
        "saqa_config": {"type": "domain_specific"},
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
            "answer_formats": [
                {
                    "pattern": r"Answer:\s*(\w+)\b",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_first",
                    "format_type": "proper",
                },
                {
                    "pattern": r"<answer>(\w+)</answer>",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_last",
                    "format_type": "improper",
                },
            ],
            "score_fns": {
                "cvss_score": {"type": "cvss"},
            },
        },
    }

    benchmark_chat = SABenchmark(
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
        "mean_cvss_score": pytest.approx(float("nan"), nan_ok=True),
        "median_cvss_score": pytest.approx(float("nan"), nan_ok=True),
        "query_count": 0,
    }

    assert [
        {
            "stats": {
                "label": "AV:A",
                "max_token_halt": False,
                "num_output_tokens": 4,
                "prediction": "AV:L",
                "answer_format": AnswerFormat.PROPER,
                "scores": {"cvss_score": {"value": 1 / 2}},
            }
        },
        {
            "stats": {
                "label": "AV:N",
                "max_token_halt": False,
                "num_output_tokens": 8,
                "prediction": "AV:N",
                "answer_format": AnswerFormat.PROPER,
                "scores": {"cvss_score": {"value": 1 / 8}},
            }
        },
        {
            "stats": {
                "label": "AV:P",
                "max_token_halt": True,
                "num_output_tokens": 12,
                "prediction": "AV:N",
                "answer_format": AnswerFormat.IMPROPER,
                "scores": {"cvss_score": {"value": 1 / 4}},
            }
        },
    ] >> metric_aggregator == {
        "format_count": {
            "improper": 1,
            "inferred": 0,
            "invalid": 0,
            "proper": 2,
        },
        "format_rate": {
            "improper": pytest.approx(1 / 3),
            "inferred": pytest.approx(0),
            "invalid": pytest.approx(0),
            "proper": pytest.approx(2 / 3),
        },
        "mean_cvss_score": pytest.approx(7 / 24),
        "mean_output_tokens": pytest.approx(8),
        "median_cvss_score": pytest.approx(1 / 4),
        "query_count": 3,
        "rate_max_token_halt": pytest.approx(1 / 3),
    }
