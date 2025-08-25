# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import cast
from unittest.mock import ANY

import pytest
from transformers import AutoTokenizer

from faith._internal.formatting import AnswerFormat
from faith._internal.types.flags import GenerationMode, SampleRatio
from faith.benchmark.benchmark import BenchmarkSpec
from faith.benchmark.categories.multiple_choice import MCBenchmark
from faith.benchmark.formatting.prompt import PromptFormatter

TEST_ROOT_DIR = Path(__file__).parent.absolute()


def test_multiple_choice_benchmark() -> None:
    benchmark = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.NEXT_TOKEN,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
        config={
            "mcqa_config": {"answer_symbols": ["A", "B"]},
            "format": {
                "instructions": {
                    "system_prompt": "You are a compassionate comptroller.",
                    "base_inst_template": "Please analyze and answer the following question.",
                    "chat_inst_template": "Please analyze and answer the following question in a chat format.",
                },
                "prompt": {
                    "question_template": """Question: {{ question }}

Choices:
{% for choice_letter, choice in choice_map.items() -%}
#{{ choice_letter }}# {{ choice }}{% if not loop.last %}
{% endif %}{% endfor -%}
""",
                    "answer_template": "Antwort--> {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
        },
    )

    assert benchmark.answer_set == {"A", "B"}
    assert benchmark.generation_mode == GenerationMode.NEXT_TOKEN


@pytest.mark.slow
def test_multiple_choice_benchmark_answer_leadin() -> None:
    benchmark = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.NEXT_TOKEN,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
        config={
            "mcqa_config": {"answer_symbols": ["A", "B"]},
            "format": {
                "instructions": {
                    "system_prompt": "You are a compassionate comptroller.",
                    "base_inst_template": "Please analyze and answer the following question.",
                    "chat_inst_template": "Please analyze and answer the following question in a chat format.",
                },
                "prompt": {
                    "question_template": """Question: {{ question }}

Choices:
{% for choice_letter, choice in choice_map.items() -%}
#{{ choice_letter }}# {{ choice }}{% if not loop.last %}
{% endif %}{% endfor -%}
""",
                    "answer_template": "Antwort--> {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
        },
    )

    tokenizer = AutoTokenizer.from_pretrained("fdtn-ai/Foundation-Sec-8B")
    assert benchmark.answer_leadin(tokenizer) == "Antwort-->"


@pytest.mark.slow
def test_multiple_choice_benchmark_answer_leadin_multiple_answer_vars() -> None:
    benchmark = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.NEXT_TOKEN,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
        config={
            "mcqa_config": {"answer_symbols": ["A", "B"]},
            "format": {
                "instructions": {
                    "system_prompt": "You are a compassionate comptroller.",
                    "base_inst_template": "Please analyze and answer the following question.",
                    "chat_inst_template": "Please analyze and answer the following question in a chat format.",
                },
                "prompt": {
                    "question_template": """Question: {{ question }}

Choices:
{% for choice_letter, choice in choice_map.items() -%}
#{{ choice_letter }}# {{ choice }}{% if not loop.last %}
{% endif %}{% endfor -%}
""",
                    "answer_template": "{{ answer }}, uhm... final answer is {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
        },
    )

    tokenizer = AutoTokenizer.from_pretrained("fdtn-ai/Foundation-Sec-8B")
    with pytest.raises(
        AssertionError, match="All pairs of answers must differ in exactly 1 token"
    ):
        benchmark.answer_leadin(tokenizer)  # noqa: B018


@pytest.mark.slow
def test_multiple_choice_benchmark_answer_token_map() -> None:
    benchmark = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.NEXT_TOKEN,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
        config={
            "mcqa_config": {"answer_symbols": ["A", "B", "C", "D", "E", "F"]},
            "format": {
                "instructions": {
                    "system_prompt": "You are a compassionate comptroller.",
                    "base_inst_template": "Please analyze and answer the following question.",
                    "chat_inst_template": "Please analyze and answer the following question in a chat format.",
                },
                "prompt": {
                    "question_template": """Question: {{ question }}

Choices:
{% for choice_letter, choice in choice_map.items() -%}
#{{ choice_letter }}# {{ choice }}{% if not loop.last %}
{% endif %}{% endfor -%}
""",
                    "answer_template": "Antwort--> {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
        },
    )

    tokenizer = AutoTokenizer.from_pretrained("fdtn-ai/Foundation-Sec-8B")
    token_map = benchmark.answer_token_map(tokenizer)
    assert token_map == {
        "A": 362,
        "B": 426,
        "C": 356,
        "D": 423,
        "E": 469,
        "F": 435,
    }
    assert {k: tokenizer.decode(v) for k, v in token_map.items()} == {
        "A": " A",
        "B": " B",
        "C": " C",
        "D": " D",
        "E": " E",
        "F": " F",
    }


def test_multiple_choice_benchmark_build_dataset() -> None:
    benchmark_1shot = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.NEXT_TOKEN,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(1),
        ),
        config={
            "mcqa_config": {"answer_symbols": ["A", "B", "C"]},
            "format": {
                "instructions": {
                    "system_prompt": "You are a compassionate comptroller.",
                    "base_inst_template": "Please analyze and answer the following question.",
                    "chat_inst_template": "Please analyze and answer the following question in a chat format.",
                },
                "prompt": {
                    "question_template": """Question: {{ question }}

Choices:
{% for choice_letter, choice in choice_map.items() -%}
#{{ choice_letter }}# {{ choice }}{% if not loop.last %}
{% endif %}{% endfor -%}
""",
                    "answer_template": "Antwort--> {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
            "source": {
                "files": {
                    "type": "csv",
                    "path_glob": "data/fake_mc_dataset.csv",
                },
            },
        },
        path=TEST_ROOT_DIR,
        seed=123,
    )
    dataset_1shot = benchmark_1shot.build_dataset()

    # Compare the questions as dictionaries.
    assert [q.to_dict() for q in dataset_1shot.iter_data()] == [
        {
            "benchmark_sample_index": 1,
            "benchmark_sample_hash": ANY,
            "subject": None,
            "system_prompt": "You are a compassionate comptroller.",
            "instruction": "Please analyze and answer the following question in a chat format.",
            "question": "What is the formula for water?",
            "choices": {"A": "H2O2", "B": "OH+", "C": "H2O"},
            "label": "C",
            "formatted_question": "Question: What is the formula for water?\n\nChoices:\n#A# H2O2\n#B# OH+\n#C# H2O",
            "formatted_answer": "Antwort--> C",
            "question_prompt": "Please analyze and answer the following question in a chat format.\n\nQuestion: What is the formula for water?\n\nChoices:\n#A# H2O2\n#B# OH+\n#C# H2O",
        }
    ]

    # Run a test on 0-shot benchmark.
    benchmark_0shot = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.NEXT_TOKEN,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
        config={
            "mcqa_config": {"answer_symbols": ["A", "B", "C", "D"]},
            "format": {
                "instructions": {
                    "system_prompt": "You are a compassionate comptroller.",
                    "base_inst_template": "Please analyze and answer the following question.",
                    "chat_inst_template": "Please analyze and answer the following question in a chat format.",
                },
                "prompt": {
                    "question_template": """Question: {{ question }}

Choices:
{% for choice_letter, choice in choice_map.items() -%}
#{{ choice_letter }}# {{ choice }}{% if not loop.last %}
{% endif %}{% endfor -%}
""",
                    "answer_template": "Antwort--> {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
            "source": {
                "files": {
                    "type": "json",
                    "path_glob": "data/*.json",
                    "selected_columns": ["questions", "metadata"],
                },
                "options": {
                    "dataframe_transform_expr": """df.assign(
    question=df["question"].str.strip(),
    answer=df["answer"].str.strip().str.upper(),
    choices=[
        [str(c) for c in lst]
        for lst in df[sorted(col for col in df.columns if col.startswith("options."))].values.tolist()
    ],
)[["question", "choices", "answer"]]""",
                },
            },
        },
        path=TEST_ROOT_DIR,
        seed=123,
    )
    dataset_0shot = benchmark_0shot.build_dataset(randomize_choices=True, sample_size=1)

    # Compare the questions as dictionaries.
    assert [q.to_dict() for q in dataset_0shot.iter_data()] == [
        {
            "benchmark_sample_index": 0,
            "benchmark_sample_hash": ANY,
            "subject": None,
            "system_prompt": "You are a compassionate comptroller.",
            "instruction": "Please analyze and answer the following question in a chat format.",
            "question": "What is the capital of Germany?",
            "choices": {"A": "Madrid", "B": "Berlin", "C": "Paris", "D": "Rome"},
            "label": "B",
            "formatted_question": "Question: What is the capital of Germany?\n\nChoices:\n#A# Madrid\n#B# Berlin\n#C# Paris\n#D# Rome",
            "formatted_answer": "Antwort--> B",
            "question_prompt": "Please analyze and answer the following question in a chat format.\n\nQuestion: What is the capital of Germany?\n\nChoices:\n#A# Madrid\n#B# Berlin\n#C# Paris\n#D# Rome",
        }
    ]


# TODO(https://github.com/RobustIntelligence/faith/issues/196): Currently the only
# public git repository we have is CyberMetric; migrate this test to use our own
# repository when it is public.
@pytest.mark.slow
def test_multiple_choice_benchmark_from_git_repo() -> None:
    # Source a benchmark from our git repository at a specific commit.
    benchmark_0shot = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.NEXT_TOKEN,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
        config={
            "mcqa_config": {"answer_symbols": ["A", "B", "C", "D"]},
            "format": {
                "instructions": {
                    "system_prompt": "You are a compassionate comptroller.",
                    "base_inst_template": "Please analyze and answer the following question.",
                    "chat_inst_template": "Please analyze and answer the following question in a chat format.",
                },
                "prompt": {
                    "question_template": """Question: {{ question }}

Choices:
{% for choice_letter, choice in choice_map.items() -%}
#{{ choice_letter }}# {{ choice }}{% if not loop.last %}
{% endif %}{% endfor -%}
""",
                    "answer_template": "Antwort--> {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
            "source": {
                "git_repo": {
                    "repo_url": "https://github.com/cybermetric/CyberMetric.git",
                    "branch": "main",
                    "commit": "2f5818bd2c19350cd6cfae028b75499ebe4ffd29",
                    "type": "json",
                    "path_glob": "CyberMetric-80-v1.json",
                    "selected_columns": ["questions"],
                },
                "options": {
                    "dataframe_transform_expr": """df.assign(
    question=df["question"].str.strip(),
    answer=df["solution"].str.strip().str.upper(),
    choices=[
        [str(c) for c in lst]
        for lst in df[
            sorted(col for col in df.columns if col.startswith("answers."))
        ].values.tolist()
    ],
)[["question", "choices", "answer"]]""",
                },
            },
        },
        seed=123,
    )
    dataset_0shot = benchmark_0shot.build_dataset(randomize_choices=True, sample_size=1)

    # Compare the questions as dictionaries.
    assert [q.to_dict() for q in dataset_0shot.iter_data()] == [
        {
            "benchmark_sample_index": 1,
            "benchmark_sample_hash": ANY,
            "subject": None,
            "system_prompt": "You are a compassionate comptroller.",
            "instruction": "Please analyze and answer the following question in a chat format.",
            "question": "In cryptography, what is the purpose of using a key-derivation function (KDF)?",
            "choices": {
                "A": "Encrypt data using a password",
                "B": "Generate public keys",
                "C": "KDF are algorithms used to transform a secret into crucial parameters like keys and Initialization Vectors (IVs)",
                "D": "Authenticate digital signatures",
            },
            "label": "C",
            "formatted_question": """Question: In cryptography, what is the purpose of using a key-derivation function (KDF)?

Choices:
#A# Encrypt data using a password
#B# Generate public keys
#C# KDF are algorithms used to transform a secret into crucial parameters like keys and Initialization Vectors (IVs)
#D# Authenticate digital signatures""",
            "formatted_answer": "Antwort--> C",
            "question_prompt": """Please analyze and answer the following question in a chat format.

Question: In cryptography, what is the purpose of using a key-derivation function (KDF)?

Choices:
#A# Encrypt data using a password
#B# Generate public keys
#C# KDF are algorithms used to transform a secret into crucial parameters like keys and Initialization Vectors (IVs)
#D# Authenticate digital signatures""",
        }
    ]


def test_multiple_choice_benchmark_log_grader() -> None:
    bench_config = {
        "mcqa_config": {"answer_symbols": ["A", "B"]},
        "format": {
            "instructions": {
                "system_prompt": "You are a compassionate comptroller.",
                "base_inst_template": "Please analyze and answer the following question.",
                "chat_inst_template": "Please analyze and answer the following question in a chat format.",
            },
            "prompt": {
                "question_template": """Question: {{ question }}

Choices:
{% for choice_letter, choice in choice_map.items() -%}
#{{ choice_letter }}# {{ choice }}{% if not loop.last %}
{% endif %}{% endfor -%}
""",
                "answer_template": "Antwort--> {{ answer }}",
                "prompt_template": "{{ instruction }}\n\n{{ question }}",
            },
        },
        "output_processing": {
            "answer_formats": [
                {
                    "pattern": r"Antwort-->\s*([A-Z])",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_first",
                    "format_type": "proper",
                },
                {
                    "pattern": r"Answer:\s+([A-Z])",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_last",
                    "format_type": "improper",
                },
            ],
        },
    }

    # Test for a benchmark with logits generation mode.
    benchmark_logits = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.LOGITS,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
        config=bench_config,
    )
    logits_log_grader = benchmark_logits.log_grader()

    assert [log["stats"] for log in [] >> logits_log_grader] == []
    assert [
        log["stats"]
        for log in [
            {
                "data": {"label": "A"},
                "model_data": {
                    "logits": [
                        [
                            {"token_id": 1, "logprob": -2.0},
                            {"token_id": 0, "logprob": -1.5},
                            {"token_id": 27, "logprob": -1.0},
                        ]
                    ],
                    "answer_symbol_ids": {"A": 1, "B": 0},
                },
            },
        ]
        >> logits_log_grader
    ] == [
        {
            "answer_format": AnswerFormat.PROPER,
            "label": "A",
            "log_probs": {
                "label": pytest.approx(-2.0),
                "max_other_symbol": pytest.approx(-1.5),
                "max_other_token": pytest.approx(-1.0),
            },
            "prediction": "B",
            "subject": None,
        }
    ]
    assert [
        log["stats"]
        for log in [
            {
                "data": {"label": "A", "subject": "frankenfood"},
                "model_data": {
                    "logits": [
                        [
                            {"token_id": 1, "logprob": -0.5},
                            {"token_id": 0, "logprob": -1.5},
                            {"token_id": 27, "logprob": -2.5},
                        ]
                    ],
                    "answer_symbol_ids": {"A": 1, "B": 0},
                },
            },
            {
                "data": {"label": "B"},
                "model_data": {
                    "logits": [
                        [
                            {"token_id": 3, "logprob": -1.0},
                            {"token_id": 0, "logprob": -1.5},
                        ]
                    ],
                    "answer_symbol_ids": {"A": 1, "B": 0},
                },
            },
            {
                "data": {"label": "A"},
                "model_data": {
                    "logits": [
                        [
                            {"token_id": 7, "logprob": -2.0},
                            {"token_id": 11, "logprob": -3.25},
                        ]
                    ],
                    "answer_symbol_ids": {"A": 1, "B": 0},
                },
            },
            {
                "data": {"label": "A"},
                "model_data": {"error": {"title": "Oopsy"}},
            },
        ]
        >> logits_log_grader
    ] == [
        {
            "answer_format": AnswerFormat.PROPER,
            "label": "A",
            "log_probs": {
                "label": pytest.approx(-0.5),
                "max_other_symbol": pytest.approx(-1.5),
                "max_other_token": pytest.approx(-1.5),
            },
            "prediction": "A",
            "subject": "frankenfood",
        },
        {
            "answer_format": AnswerFormat.PROPER,
            "label": "B",
            "log_probs": {
                "label": pytest.approx(-1.5),
                "max_other_symbol": pytest.approx(float("-inf")),
                "max_other_token": pytest.approx(-1.0),
            },
            "prediction": "B",
            "subject": None,
        },
        {
            "answer_format": AnswerFormat.INVALID,
            "label": "A",
            "log_probs": {
                "label": pytest.approx(float("-inf")),
                "max_other_symbol": pytest.approx(float("-inf")),
                "max_other_token": pytest.approx(-2.0),
            },
            "prediction": None,
            "subject": None,
        },
        {
            "answer_format": AnswerFormat.INVALID,
            "label": "A",
            "prediction": None,
            "subject": None,
        },
    ]

    # Test for a benchmark with next token generation mode.
    benchmark_next_token = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.NEXT_TOKEN,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
        config=bench_config,
    )
    next_token_log_grader = benchmark_next_token.log_grader(recompute_stats=True)

    assert [log["stats"] for log in [] >> next_token_log_grader] == []
    assert [
        log["stats"]
        for log in [
            {
                "data": {"label": "A"},
                "model_data": {"next_token": {"output_text": " A"}},
            },
        ]
        >> next_token_log_grader
    ] == [
        {
            "answer_format": AnswerFormat.PROPER,
            "label": "A",
            "prediction": "A",
            "subject": None,
        }
    ]
    assert [
        log["stats"]
        for log in cast(
            list[dict],
            [
                {
                    "data": {"label": "B", "subject": "octothorpes"},
                    "model_data": {"next_token": {"output_text": "B"}},
                    "stats": {
                        "label": "B",
                        "prediction": "C",
                        "answer_format": AnswerFormat.IMPROPER,
                    },
                },
                {
                    "data": {"label": "A"},
                    "model_data": {"next_token": {"output_text": " B"}},
                },
                {
                    "data": {"label": "B"},
                    "model_data": {"next_token": {"output_text": "B or A"}},
                },
                {
                    "data": {"label": "A", "subject": "octothorpes"},
                    "model_data": {"next_token": {"output_text": "I don't know"}},
                },
                {
                    "data": {"label": "A", "subject": "octothorpes"},
                    "model_data": {"error": {"title": "Oopie"}},
                },
            ],
        )
        >> next_token_log_grader
    ] == [
        {
            "answer_format": AnswerFormat.PROPER,
            "label": "B",
            "prediction": "B",
            "subject": "octothorpes",
        },
        {
            "answer_format": AnswerFormat.PROPER,
            "label": "A",
            "prediction": "B",
            "subject": None,
        },
        {
            "answer_format": AnswerFormat.PROPER,
            "label": "B",
            "prediction": "B",
            "subject": None,
        },
        {
            "answer_format": AnswerFormat.INVALID,
            "label": "A",
            "prediction": None,
            "subject": "octothorpes",
        },
        {
            "answer_format": AnswerFormat.INVALID,
            "label": "A",
            "prediction": None,
            "subject": "octothorpes",
        },
    ]

    # Test for a benchmark with chat-completion generation mode.
    benchmark_chat = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
        config=bench_config,
    )
    chat_log_grader = benchmark_chat.log_grader()

    assert [log["stats"] for log in [] >> chat_log_grader] == []
    assert [
        log["stats"]
        for log in cast(
            list[dict],
            [
                {
                    "data": {"label": "A"},
                    "model_data": {
                        "chat_comp": {
                            "output_text": "Antwort--> A",
                            "num_output_tokens": 3,
                            "max_token_halt": False,
                        }
                    },
                },
                {
                    "stats": {
                        "answer_format": "improper",
                        "label": "B",
                        "prediction": "C",
                    }
                },
                {
                    "data": {"label": "B"},
                    "model_data": {
                        "chat_comp": {
                            "output_text": "uhm... I have no earthly idea",
                            "num_output_tokens": 7,
                            "max_token_halt": True,
                        },
                    },
                },
            ],
        )
        >> chat_log_grader
    ] == [
        {
            "answer_format": AnswerFormat.PROPER,
            "label": "A",
            "max_token_halt": False,
            "num_output_tokens": 3,
            "prediction": "A",
            "subject": None,
        },
        {
            "answer_format": AnswerFormat.IMPROPER,
            "label": "B",
            "prediction": "C",
        },
        {
            "answer_format": AnswerFormat.INVALID,
            "label": "B",
            "max_token_halt": True,
            "num_output_tokens": 7,
            "prediction": None,
            "subject": None,
        },
    ]
    assert [
        log["stats"]
        for log in [
            {
                "data": {"label": "A"},
                "model_data": {
                    "chat_comp": {
                        "output_text": "Antwort--> A",
                        "num_output_tokens": 4,
                        "max_token_halt": False,
                    }
                },
            },
            {
                "data": {"label": "B"},
                "model_data": {
                    "chat_comp": {
                        "output_text": "I think the answer is B or A. Guessing...\n\nAnswer: A",
                        "num_output_tokens": 17,
                        "max_token_halt": True,
                    }
                },
            },
            {
                "data": {"label": "A"},
                "model_data": {
                    "chat_comp": {
                        "output_text": "If I had to guess, I would say A",
                        "num_output_tokens": 10,
                        "max_token_halt": False,
                    },
                },
            },
            {
                "data": {"label": "A"},
                "model_data": {"error": {"title": "Oops"}},
            },
        ]
        >> chat_log_grader
    ] == [
        {
            "answer_format": AnswerFormat.PROPER,
            "label": "A",
            "max_token_halt": False,
            "num_output_tokens": 4,
            "prediction": "A",
            "subject": None,
        },
        {
            "answer_format": AnswerFormat.IMPROPER,
            "label": "B",
            "max_token_halt": True,
            "num_output_tokens": 17,
            "prediction": "A",
            "subject": None,
        },
        {
            "answer_format": AnswerFormat.INVALID,
            "label": "A",
            "max_token_halt": False,
            "num_output_tokens": 10,
            "prediction": None,
            "subject": None,
        },
        {
            "answer_format": AnswerFormat.INVALID,
            "label": "A",
            "max_token_halt": False,
            "num_output_tokens": 0,
            "prediction": None,
            "subject": None,
        },
    ]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_multiple_choice_benchmark_grade_aggregator_logits() -> None:
    bench_config = {
        "mcqa_config": {"answer_symbols": ["A", "B"]},
        "format": {
            "instructions": {
                "system_prompt": "You are a compassionate comptroller.",
                "base_inst_template": "Please analyze and answer the following question.",
                "chat_inst_template": "Please analyze and answer the following question in a chat format.",
            },
            "prompt": {
                "question_template": """Question: {{ question }}

Choices:
{% for choice_letter, choice in choice_map.items() -%}
#{{ choice_letter }}# {{ choice }}{% if not loop.last %}
{% endif %}{% endfor -%}
""",
                "answer_template": "Antwort--> {{ answer }}",
                "prompt_template": "{{ instruction }}\n\n{{ question }}",
            },
        },
        "output_processing": {
            "answer_formats": [
                {
                    "pattern": r"Antwort-->\s*([A-Z])",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_first",
                    "format_type": "proper",
                },
                {
                    "pattern": r"Answer:\s+([A-Z])",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_last",
                    "format_type": "improper",
                },
            ],
        },
    }

    benchmark_chat = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.LOGITS,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
        config=bench_config,
    )
    metric_aggregator = benchmark_chat.grade_aggregator()

    assert [] >> metric_aggregator == {
        "accuracy": pytest.approx(float("nan"), nan_ok=True),
        "confusion_matrix_count": {
            "A": {"A": 0, "B": 0, "": 0},
            "B": {"A": 0, "B": 0, "": 0},
            "": {"A": 0, "B": 0, "": 0},
        },
        "f1_scores": {
            "A": pytest.approx(float("nan"), nan_ok=True),
            "B": pytest.approx(float("nan"), nan_ok=True),
        },
        "format_breakdown_count": {
            "improper": {"correct": 0, "incorrect": 0},
            "inferred": {"correct": 0, "incorrect": 0},
            "invalid": {"correct": 0, "incorrect": 0},
            "proper": {"correct": 0, "incorrect": 0},
        },
        "format_count": {"improper": 0, "inferred": 0, "invalid": 0, "proper": 0},
        "format_rate": {
            "improper": pytest.approx(float("nan"), nan_ok=True),
            "inferred": pytest.approx(float("nan"), nan_ok=True),
            "invalid": pytest.approx(float("nan"), nan_ok=True),
            "proper": pytest.approx(float("nan"), nan_ok=True),
        },
        "label_count": {"A": 0, "B": 0},
        "lenient_accuracy": pytest.approx(float("nan"), nan_ok=True),
        "query_count": 0,
        "weighted_avg_f1": pytest.approx(float("nan"), nan_ok=True),
    }

    assert cast(
        list[dict],
        [
            {
                "stats": {
                    "label": "A",
                    "log_probs": {
                        "label": -2.0,
                        "max_other_symbol": -1.5,
                        "max_other_token": -1.0,
                    },
                    "prediction": "B",
                    "answer_format": AnswerFormat.PROPER,
                    "subject": "bumbershoots",
                }
            },
            {
                "stats": {
                    "label": "B",
                    "log_probs": {
                        "label": -0.5,
                        "max_other_symbol": -1.5,
                        "max_other_token": -1.0,
                    },
                    "prediction": "B",
                    "answer_format": AnswerFormat.PROPER,
                    "subject": "bumbershoots",
                }
            },
            {
                "stats": {
                    "label": "B",
                    "log_probs": {
                        "label": -1.5,
                        "max_other_symbol": -1.0,
                        "max_other_token": float("-inf"),
                    },
                    "prediction": "A",
                    "answer_format": AnswerFormat.IMPROPER,
                    "subject": "blabberdash",
                }
            },
            {
                "stats": {
                    "label": "A",
                    "log_probs": {
                        "label": float("-inf"),
                        "max_other_symbol": float("-inf"),
                        "max_other_token": -0.25,
                    },
                    "prediction": None,
                    "answer_format": AnswerFormat.INVALID,
                    "subject": "blabberdash",
                }
            },
        ],
    ) >> metric_aggregator == {
        "accuracy": pytest.approx(1 / 4),
        "accuracy_per_subject": {
            "bumbershoots": pytest.approx(1 / 2),
            "blabberdash": pytest.approx(0),
        },
        "confusion_matrix_count": {
            "A": {"A": 0, "B": 1, "": 1},
            "B": {"A": 1, "B": 1, "": 0},
            "": {"A": 0, "B": 0, "": 0},
        },
        "f1_scores": {"A": pytest.approx(0), "B": pytest.approx(1 / 2)},
        "format_breakdown_count": {
            "improper": {"correct": 0, "incorrect": 1},
            "inferred": {"correct": 0, "incorrect": 0},
            "invalid": {"correct": 0, "incorrect": 1},
            "proper": {"correct": 1, "incorrect": 1},
        },
        "format_count": {"improper": 1, "inferred": 0, "invalid": 1, "proper": 2},
        "format_rate": {
            "improper": pytest.approx(1 / 4),
            "inferred": pytest.approx(0),
            "invalid": pytest.approx(1 / 4),
            "proper": pytest.approx(1 / 2),
        },
        "label_count": {"A": 2, "B": 2},
        "lenient_accuracy": pytest.approx(1 / 4),
        "lenient_accuracy_per_subject": {
            "bumbershoots": pytest.approx(1 / 2),
            "blabberdash": pytest.approx(0),
        },
        "mean_confidence_gap": {
            "correct_symbol": 1.0,
            "correct_token": 0.5,
            "incorrect_symbol": 0.5,
            "incorrect_token": 1.0,
        },
        "query_count": 4,
        "num_near_ties": {
            "symbol": 0,
            "token": 0,
        },
        "perplexity": pytest.approx(3.7936678946831774),
        "subject_weighted_accuracy": pytest.approx(1 / 4),
        "subject_weighted_lenient_accuracy": pytest.approx(1 / 4),
        "weighted_avg_f1": pytest.approx(1 / 4),
    }


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_multiple_choice_benchmark_grade_aggregator_chat() -> None:
    bench_config = {
        "mcqa_config": {"answer_symbols": ["A", "B"]},
        "format": {
            "instructions": {
                "system_prompt": "You are a compassionate comptroller.",
                "base_inst_template": "Please analyze and answer the following question.",
                "chat_inst_template": "Please analyze and answer the following question in a chat format.",
            },
            "prompt": {
                "question_template": """Question: {{ question }}

Choices:
{% for choice_letter, choice in choice_map.items() -%}
#{{ choice_letter }}# {{ choice }}{% if not loop.last %}
{% endif %}{% endfor -%}
""",
                "answer_template": "Antwort--> {{ answer }}",
                "prompt_template": "{{ instruction }}\n\n{{ question }}",
            },
        },
        "output_processing": {
            "answer_formats": [
                {
                    "pattern": r"Antwort-->\s*([A-Z])",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_first",
                    "format_type": "proper",
                },
                {
                    "pattern": r"Answer:\s+([A-Z])",
                    "capture_transform": {"params": ["x"], "expr": "x.strip().upper()"},
                    "match_disambiguation": "match_last",
                    "format_type": "improper",
                },
            ],
        },
    }

    benchmark_chat = MCBenchmark(
        spec=BenchmarkSpec(
            name="test-bar",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
        config=bench_config,
    )
    metric_aggregator = benchmark_chat.grade_aggregator()

    assert [] >> metric_aggregator == {
        "accuracy": pytest.approx(float("nan"), nan_ok=True),
        "confusion_matrix_count": {
            "A": {"A": 0, "B": 0, "": 0},
            "B": {"A": 0, "B": 0, "": 0},
            "": {"A": 0, "B": 0, "": 0},
        },
        "f1_scores": {
            "A": pytest.approx(float("nan"), nan_ok=True),
            "B": pytest.approx(float("nan"), nan_ok=True),
        },
        "format_breakdown_count": {
            "improper": {"correct": 0, "incorrect": 0},
            "inferred": {"correct": 0, "incorrect": 0},
            "invalid": {"correct": 0, "incorrect": 0},
            "proper": {"correct": 0, "incorrect": 0},
        },
        "format_count": {"improper": 0, "inferred": 0, "invalid": 0, "proper": 0},
        "format_rate": {
            "improper": pytest.approx(float("nan"), nan_ok=True),
            "inferred": pytest.approx(float("nan"), nan_ok=True),
            "invalid": pytest.approx(float("nan"), nan_ok=True),
            "proper": pytest.approx(float("nan"), nan_ok=True),
        },
        "label_count": {"A": 0, "B": 0},
        "lenient_accuracy": pytest.approx(float("nan"), nan_ok=True),
        "query_count": 0,
        "weighted_avg_f1": pytest.approx(float("nan"), nan_ok=True),
    }

    assert [
        {
            "stats": {
                "label": "A",
                "max_token_halt": False,
                "num_output_tokens": 3,
                "prediction": "B",
                "answer_format": AnswerFormat.PROPER,
                "subject": "bumbershoots",
            }
        },
        {
            "stats": {
                "label": "B",
                "max_token_halt": False,
                "num_output_tokens": 41,
                "prediction": "B",
                "answer_format": AnswerFormat.PROPER,
                "subject": "bumbershoots",
            }
        },
        {
            "stats": {
                "label": "B",
                "max_token_halt": False,
                "num_output_tokens": 4,
                "prediction": "A",
                "answer_format": AnswerFormat.IMPROPER,
                "subject": "blabberdash",
            }
        },
    ] >> metric_aggregator == {
        "accuracy": pytest.approx(1 / 3),
        "accuracy_per_subject": {
            "bumbershoots": pytest.approx(1 / 2),
            "blabberdash": pytest.approx(0),
        },
        "confusion_matrix_count": {
            "A": {"A": 0, "B": 1, "": 0},
            "B": {"A": 1, "B": 1, "": 0},
            "": {"A": 0, "B": 0, "": 0},
        },
        "f1_scores": {"A": pytest.approx(0), "B": pytest.approx(1 / 2)},
        "format_breakdown_count": {
            "improper": {"correct": 0, "incorrect": 1},
            "inferred": {"correct": 0, "incorrect": 0},
            "invalid": {"correct": 0, "incorrect": 0},
            "proper": {"correct": 1, "incorrect": 1},
        },
        "format_count": {"improper": 1, "inferred": 0, "invalid": 0, "proper": 2},
        "format_rate": {
            "improper": pytest.approx(1 / 3),
            "inferred": pytest.approx(0),
            "invalid": pytest.approx(0),
            "proper": pytest.approx(2 / 3),
        },
        "label_count": {"A": 1, "B": 2},
        "lenient_accuracy": pytest.approx(1 / 3),
        "lenient_accuracy_per_subject": {
            "bumbershoots": pytest.approx(1 / 2),
            "blabberdash": pytest.approx(0),
        },
        "mean_output_tokens": pytest.approx(16),
        "query_count": 3,
        "rate_max_token_halt": pytest.approx(0),
        "subject_weighted_accuracy": pytest.approx(1 / 4),
        "subject_weighted_lenient_accuracy": pytest.approx(1 / 4),
        "weighted_avg_f1": pytest.approx(1 / 3),
    }
