# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import ANY, patch

import pytest
from datasets import Dataset, DatasetDict, Features, Value

from faith import __version__
from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.benchmark.spec import BenchmarkSpec
from faith._types.config.benchmark import BenchmarkConfig, SAQAConfig, ShortAnswerType
from faith._types.config.format import FormatConfig, InstructionsConfig, PromptConfig
from faith._types.config.patterns import (
    AnswerFormat,
    CaptureTransform,
    Disambiguation,
    PatternDef,
)
from faith._types.config.scoring import OutputProcessingConfig, ScoreFnConfig
from faith._types.config.source import HuggingFaceSourceConfig, SourceConfig
from faith._types.model.generation import GenerationMode
from faith._types.model.prompt import PromptFormatter
from faith._types.record.stats import StatsRecord
from faith.benchmark.categories.short_answer import SABenchmark
from tests.benchmark.categories.fake_record_maker import make_fake_record


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
            config=BenchmarkConfig(
                saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH),
                format=FormatConfig(
                    instructions=InstructionsConfig(
                        system_prompt_template="You are a helpful assistant.",
                        base_inst_template="Please answer the following question.",
                        chat_inst_template="Please answer the following question in a chat format.",
                    ),
                    prompt=PromptConfig(
                        question_template="Question: {{ question }}",
                        answer_template="Answer: {{ answer }}",
                        prompt_template="{{ instruction }}\n\n{{ question }}",
                    ),
                ),
            ),
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
            config=BenchmarkConfig(
                saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH),
                format=FormatConfig(
                    instructions=InstructionsConfig(
                        system_prompt_template="You are a helpful assistant.",
                        base_inst_template="Please answer the following question.",
                        chat_inst_template="Please answer the following question in a chat format.",
                    ),
                    prompt=PromptConfig(
                        question_template="Question: {{ question }}",
                        answer_template="Answer: {{ answer }}",
                        prompt_template="{{ instruction }}\n\n{{ question }}",
                    ),
                ),
            ),
        )


def test_short_answer_benchmark_chat() -> None:
    benchmark = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(5),
        ),
        config=BenchmarkConfig(
            saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH),
            format=FormatConfig(
                instructions=InstructionsConfig(
                    base_inst_template="Please answer the following question.",
                    chat_inst_template="Please answer the following question in a chat format.",
                ),
                prompt=PromptConfig(
                    question_template="Question: {{ question }}",
                    answer_template="Answer: {{ answer }}",
                    prompt_template="{{ instruction }}\n\n{{ question }}",
                ),
            ),
        ),
    )

    assert benchmark.answer_set is None
    assert benchmark.generation_mode == GenerationMode.CHAT_COMP
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
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(1),
        ),
        config=BenchmarkConfig(
            saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH),
            format=FormatConfig(
                instructions=InstructionsConfig(
                    system_prompt_template="You are a helpful assistant.",
                    base_inst_template="Please answer the following question.",
                    chat_inst_template="Please answer the following question in a chat format.",
                ),
                prompt=PromptConfig(
                    question_template="Question: {{ question }}",
                    answer_template="Answer: {{ answer }}",
                    prompt_template="{{ instruction }}\n\n{{ question }}",
                ),
            ),
            source=SourceConfig(
                huggingface=HuggingFaceSourceConfig(
                    path="foo/bar-baz",
                    subset_name="qux",
                    test_split="test",
                    dev_split="dev",
                ),
            ),
        ),
        seed=42,
    )
    with patch(
        "faith.benchmark.dataset.load.load_dataset",
        return_value=fake_dataset_dict,
    ) as mock_load_dataset:
        dataset_1shot = benchmark_1shot.build_dataset()
        mock_load_dataset.assert_called_once_with("foo/bar-baz", "qux")

        # Compare the questions as dictionaries.
        assert [rec.to_dict() for rec in dataset_1shot.iter_data()] == [
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
            },
        ]

    benchmark_1shot_no_dev = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(1),
        ),
        config=BenchmarkConfig(
            saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH),
            format=FormatConfig(
                instructions=InstructionsConfig(
                    system_prompt_template="You are a helpful assistant.",
                    base_inst_template="Please answer the following question.",
                    chat_inst_template="Please answer the following question in a chat format.",
                ),
                prompt=PromptConfig(
                    question_template="Question: {{ question }}",
                    answer_template="Answer: {{ answer }}",
                    prompt_template="{{ instruction }}\n\n{{ question }}",
                ),
            ),
            source=SourceConfig(
                huggingface=HuggingFaceSourceConfig(
                    path="foo/bar-baz",
                    subset_name="qux",
                    test_split="test",
                ),
            ),
        ),
        seed=42,
    )
    with patch(
        "faith.benchmark.dataset.load.load_dataset",
        return_value=fake_dataset_dict,
    ) as mock_load_dataset:
        dataset_1shot_no_dev = benchmark_1shot_no_dev.build_dataset()
        mock_load_dataset.assert_called_once_with("foo/bar-baz", "qux")

        # Compare the questions as dictionaries.
        assert [rec.to_dict() for rec in dataset_1shot_no_dev.iter_data()] == [
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
            },
        ]

    benchmark_0shot = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(0),
        ),
        config=BenchmarkConfig(
            saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH),
            format=FormatConfig(
                instructions=InstructionsConfig(
                    system_prompt_template="You are a helpful assistant.",
                    base_inst_template="Please answer the following question.",
                    chat_inst_template="Please answer the following question in a chat format.",
                ),
                prompt=PromptConfig(
                    question_template="Question: {{ question }}",
                    answer_template="Answer: {{ answer }}",
                    prompt_template="{{ instruction }}\n\n{{ question }}",
                ),
            ),
            source=SourceConfig(
                huggingface=HuggingFaceSourceConfig(
                    path="foo/bar-baz",
                    test_split="test",
                ),
            ),
        ),
        seed=42,
    )
    with patch(
        "faith.benchmark.dataset.load.load_dataset",
        return_value=fake_dataset_dict,
    ) as mock_load_dataset:
        dataset_0shot = benchmark_0shot.build_dataset(sample_size=1)
        mock_load_dataset.assert_called_once_with("foo/bar-baz", None)

        # Compare the questions as dictionaries.
        assert [rec.to_dict() for rec in dataset_0shot.iter_data()] == [
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
            },
        ]


def test_short_answer_benchmark_process_logs_chat() -> None:
    bench_config = BenchmarkConfig(
        saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH),
        format=FormatConfig(
            instructions=InstructionsConfig(
                system_prompt_template="You are a helpful assistant.",
                base_inst_template="Please answer the following question.",
                chat_inst_template="Please answer the following question in a chat format.",
            ),
            prompt=PromptConfig(
                question_template="Question: {{ question }}",
                answer_template="Answer: {{ answer }}",
                prompt_template="{{ instruction }}\n\n{{ question }}",
            ),
        ),
        output_processing=OutputProcessingConfig(
            answer_formats=[
                PatternDef(
                    pattern=r"Answer:\s*(\w+)\b",
                    capture_transform=CaptureTransform(
                        params=["x"], expr="x.strip().lower()"
                    ),
                    disambiguation=Disambiguation.MATCH_FIRST,
                    format_type=AnswerFormat.PROPER,
                ),
                PatternDef(
                    pattern=r"<answer>(\w+)</answer>",
                    capture_transform=CaptureTransform(
                        params=["x"], expr="x.strip().lower()"
                    ),
                    disambiguation=Disambiguation.MATCH_LAST,
                    format_type=AnswerFormat.IMPROPER,
                ),
            ],
        ),
    )

    benchmark_chat = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(5),
        ),
        config=bench_config,
    )
    log_grader = benchmark_chat.log_grader(
        model_format_config=PatternDef(
            pattern=r"(?s)(?:\s*<t>.*</t>(?!.*</t>)|\s*<t>.*)?(.*)",
            disambiguation=Disambiguation.MATCH_ALL,
            format_type=AnswerFormat.PROPER,
        )
    )

    assert [log.stats for log in [] >> log_grader] == []
    assert [
        log.stats
        for log in [
            make_fake_record(
                data={"label": "foo", "subject": "bar"},
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
                data={"label": "foo", "subject": "bar"},
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
                data={"label": "bar"},
                model_data={
                    "chat_comp": {
                        "answer_text": "<answer>BaZ</answer>",
                        "output_text": "<answer>BaZ</answer>",
                        "num_output_tokens": 5,
                        "max_token_halt": False,
                    }
                },
            ),
            make_fake_record(
                data={"label": "baz"},
                model_data={
                    "chat_comp": {
                        "answer_text": "uhm... I have no earthly idea",
                        "output_text": "uhm... I have no earthly idea",
                        "num_output_tokens": 8,
                        "max_token_halt": True,
                    },
                },
            ),
            make_fake_record(
                data={"label": "qux"},
                model_data={"error": {"title": "No response from model"}},
            ),
            make_fake_record(
                stats=StatsRecord(
                    label="aaa",
                    prediction="b",
                    answer_format=AnswerFormat.PROPER,
                    subject="esperanto",
                ),
            ),
        ]
        >> log_grader
    ] == [
        StatsRecord(
            label="foo",
            prediction="foo",
            answer_format=AnswerFormat.PROPER,
            subject="bar",
            num_output_tokens=3,
            max_token_halt=False,
        ),
        StatsRecord(
            label="foo",
            prediction=None,
            answer_format=AnswerFormat.INVALID,
            subject="bar",
            num_output_tokens=3,
            max_token_halt=True,
        ),
        StatsRecord(
            label="bar",
            prediction="baz",
            answer_format=AnswerFormat.IMPROPER,
            num_output_tokens=5,
            max_token_halt=False,
        ),
        StatsRecord(
            label="baz",
            prediction=None,
            answer_format=AnswerFormat.INVALID,
            num_output_tokens=8,
            max_token_halt=True,
        ),
        StatsRecord(
            label="qux",
            prediction=None,
            answer_format=AnswerFormat.INVALID,
            num_output_tokens=0,
            max_token_halt=False,
        ),
        StatsRecord(
            label="aaa",
            prediction="b",
            answer_format=AnswerFormat.PROPER,
            subject="esperanto",
        ),
    ]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_short_answer_benchmark_grade_aggregator_string_match() -> None:
    bench_config = BenchmarkConfig(
        saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH),
        format=FormatConfig(
            instructions=InstructionsConfig(
                system_prompt_template="You are a helpful assistant.",
                base_inst_template="Please answer the following question.",
                chat_inst_template="Please answer the following question in a chat format.",
            ),
            prompt=PromptConfig(
                question_template="Question: {{ question }}",
                answer_template="Answer: {{ answer }}",
                prompt_template="{{ instruction }}\n\n{{ question }}",
            ),
        ),
        output_processing=OutputProcessingConfig(
            answer_formats=[
                PatternDef(
                    pattern=r"Answer:\s*(\w+)\b",
                    capture_transform=CaptureTransform(
                        params=["x"], expr="x.strip().upper()"
                    ),
                    disambiguation=Disambiguation.MATCH_FIRST,
                    format_type=AnswerFormat.PROPER,
                ),
                PatternDef(
                    pattern=r"<answer>(\w+)</answer>",
                    capture_transform=CaptureTransform(
                        params=["x"], expr="x.strip().upper()"
                    ),
                    disambiguation=Disambiguation.MATCH_LAST,
                    format_type=AnswerFormat.IMPROPER,
                ),
            ],
        ),
    )

    benchmark_chat = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
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
        StatsRecord(
            label="A",
            prediction="A",
            answer_format=AnswerFormat.PROPER,
            subject="esperanto",
            num_output_tokens=10,
            max_token_halt=True,
        ),
        StatsRecord(
            label="B",
            prediction="C",
            answer_format=AnswerFormat.PROPER,
            subject="esperanto",
            num_output_tokens=10,
            max_token_halt=True,
        ),
        StatsRecord(
            label="B",
            prediction="B",
            answer_format=AnswerFormat.IMPROPER,
            subject="esperanto",
            num_output_tokens=1,
            max_token_halt=False,
        ),
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
    bench_config = BenchmarkConfig(
        saqa_config=SAQAConfig(type=ShortAnswerType.LABEL_SET),
        format=FormatConfig(
            instructions=InstructionsConfig(
                system_prompt_template="You are a helpful assistant.",
                base_inst_template="Please answer the following question.",
                chat_inst_template="Please answer the following question in a chat format.",
            ),
            prompt=PromptConfig(
                question_template="Question: {{ question }}",
                answer_template="Answer: {{ answer }}",
                prompt_template="{{ instruction }}\n\n{{ question }}",
            ),
        ),
        output_processing=OutputProcessingConfig(
            answer_formats=[
                PatternDef(
                    pattern=r"Answer:\s*(\w+)\b",
                    capture_transform=CaptureTransform(
                        params=["x"], expr="x.strip().upper()"
                    ),
                    disambiguation=Disambiguation.MATCH_FIRST,
                    format_type=AnswerFormat.PROPER,
                ),
                PatternDef(
                    pattern=r"<answer>(\w+)</answer>",
                    capture_transform=CaptureTransform(
                        params=["x"], expr="x.strip().upper()"
                    ),
                    disambiguation=Disambiguation.MATCH_LAST,
                    format_type=AnswerFormat.IMPROPER,
                ),
            ],
            score_fns={
                "jaccard_index": ScoreFnConfig(type="jaccard"),
            },
        ),
    )

    benchmark_chat = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
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
        StatsRecord(
            label=["A", "B"],
            prediction=["B", "C"],
            answer_format=AnswerFormat.PROPER,
            num_output_tokens=13,
            max_token_halt=False,
            scores={"jaccard_index": {"value": 1 / 3}},
        ),
        StatsRecord(
            label=["B"],
            prediction=["B"],
            answer_format=AnswerFormat.PROPER,
            num_output_tokens=17,
            max_token_halt=True,
            scores={"jaccard_index": {"value": 1 / 2}},
        ),
        StatsRecord(
            label=["B", "C"],
            prediction=["A"],
            answer_format=AnswerFormat.IMPROPER,
            num_output_tokens=9,
            max_token_halt=False,
            scores={"jaccard_index": {"value": 0}},
        ),
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
    bench_config = BenchmarkConfig(
        saqa_config=SAQAConfig(type=ShortAnswerType.DOMAIN_SPECIFIC),
        format=FormatConfig(
            instructions=InstructionsConfig(
                system_prompt_template="You are a helpful assistant.",
                base_inst_template="Please answer the following question.",
                chat_inst_template="Please answer the following question in a chat format.",
            ),
            prompt=PromptConfig(
                question_template="Question: {{ question }}",
                answer_template="Answer: {{ answer }}",
                prompt_template="{{ instruction }}\n\n{{ question }}",
            ),
        ),
        output_processing=OutputProcessingConfig(
            answer_formats=[
                PatternDef(
                    pattern=r"Answer:\s*(\w+)\b",
                    capture_transform=CaptureTransform(
                        params=["x"], expr="x.strip().upper()"
                    ),
                    disambiguation=Disambiguation.MATCH_FIRST,
                    format_type=AnswerFormat.PROPER,
                ),
                PatternDef(
                    pattern=r"<answer>(\w+)</answer>",
                    capture_transform=CaptureTransform(
                        params=["x"], expr="x.strip().upper()"
                    ),
                    disambiguation=Disambiguation.MATCH_LAST,
                    format_type=AnswerFormat.IMPROPER,
                ),
            ],
            score_fns={
                "cvss_score": ScoreFnConfig(type="cvss"),
            },
        ),
    )

    benchmark_chat = SABenchmark(
        spec=BenchmarkSpec(
            name="test-foo",
            generation_mode=GenerationMode.CHAT_COMP,
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
        StatsRecord(
            label="AV:A",
            prediction="AV:L",
            answer_format=AnswerFormat.PROPER,
            num_output_tokens=4,
            max_token_halt=False,
            scores={"cvss_score": {"value": 1 / 2}},
        ),
        StatsRecord(
            label="AV:N",
            prediction="AV:N",
            answer_format=AnswerFormat.PROPER,
            num_output_tokens=8,
            max_token_halt=False,
            scores={"cvss_score": {"value": 1 / 8}},
        ),
        StatsRecord(
            label="AV:P",
            prediction="AV:N",
            answer_format=AnswerFormat.IMPROPER,
            num_output_tokens=12,
            max_token_halt=True,
            scores={"cvss_score": {"value": 1 / 4}},
        ),
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
