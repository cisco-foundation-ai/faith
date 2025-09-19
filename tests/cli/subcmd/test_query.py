# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence, cast
from unittest.mock import ANY

import numpy as np
import pandas as pd
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from faith._internal.algo.sampling import NShotSampler
from faith._internal.io.json import write_as_json
from faith._internal.types.flags import GenerationMode, SampleRatio
from faith.benchmark.benchmark import Benchmark, BenchmarkDataset
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.benchmark.formatting.qa import QAFormatter, QARecord
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader
from faith.benchmark.types import BenchmarkSpec
from faith.cli.subcmd.query import DataSamplingParams, GenParams, query_over_benchmark
from faith.model.base import (
    BaseModel,
    ChatResponse,
    GenerationError,
    PromptList,
    TokenPred,
)


class FakeBenchmark(Benchmark):
    def __init__(self, spec: BenchmarkSpec, config: dict[str, Any], **kwargs: Any):
        super().__init__(spec, config, **kwargs)

    def answer_leadin(self, tokenizer: PreTrainedTokenizerBase) -> str:
        return "A: "

    def answer_token_map(self, _tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
        """Return a mapping of answer tokens to their IDs."""
        return {"A": 87, "B": 31, "C": 7, "D": 9, "E": 5}

    def build_dataset(
        self,
        sample_size: int | None = None,
        randomize_choices: bool = False,
    ) -> BenchmarkDataset:
        return FakeDataset(self.formatter, np.random.default_rng())

    def log_grader(self, recompute_stats: bool = False) -> LogGrader:
        raise NotImplementedError("This method should not be called.")

    def grade_aggregator(self) -> GradeAggregator:
        raise NotImplementedError("This method should not be called.")


class FakeDataset(BenchmarkDataset):
    def __init__(self, formatter: QAFormatter, rng: np.random.Generator):
        super().__init__(
            formatter,
            # Generate fake data for testing purposes
            pd.DataFrame(
                {
                    "question": ["What is 0?", "What is 1?"],
                    "answer": ["0", "1"],
                }
            ),
            NShotSampler(None, SampleRatio(0), rng),
            rng,
            required_columns=frozenset({"question", "answer"}),
        )

    def _format_qa(
        self, index: int, sample: pd.Series, examples: Sequence[QARecord] | None = None
    ) -> QARecord:
        return self._formatter.render_qa_record(
            index=index,
            sample_hash=f"f{index}",
            raw_question=sample["question"],
            raw_answer=sample["answer"],
            examples=examples,
            subject="apiculture",
        )


class FakeModel(BaseModel):
    """A fake model that simulates responses for testing purposes."""

    def __init__(self, model_name: str):
        super().__init__(model_name)

        # Create a fake tokenizer for testing purposes.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            write_as_json(
                tmp_path / "tokenizer_config.json",
                {
                    "added_tokens_decoder": {},
                    "bos_token": "<|begin_of_text|>",
                    "chat_template": "{%- if add_bos_token %}{{- bos_token }}{%- endif %}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n<think>\n' }}{% endif %}{% endfor %}",
                    "clean_up_tokenization_spaces": True,
                    "eos_token": "<|end_of_text|>",
                    "tokenizer_class": "PreTrainedTokenizerFast",
                },
            )
            write_as_json(
                tmp_path / "tokenizer.json",
                {
                    "model": {
                        "vocab": {},
                        "merges": [],
                        "ignore_merges": True,
                        "type": "BPE",
                    },
                },
            )
            self._tokenizer = AutoTokenizer.from_pretrained(tmp_path)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        """Return the tokenizer for the model."""
        return self._tokenizer

    @property
    def supported_formats(self) -> set[PromptFormatter]:
        return set(list(PromptFormatter))

    def query(
        self,
        inputs: PromptList,
        **kwargs: Any,
    ) -> Iterable[ChatResponse | GenerationError]:
        # Simulate a fake response for testing purposes
        return [
            ChatResponse(
                prompt_token_ids=None,
                num_prompt_tokens=len(input_str),
                prompt_text=input_str,
                output_token_ids=None,
                num_output_tokens=17,
                output_text=f"Fake response to: {input_str}",
                max_token_halt=False,
            )
            for input_ in inputs
            if (input_str := cast(str, input_)) is not None
        ]

    def next_token(
        self,
        inputs: PromptList,
        **kwargs: Any,
    ) -> Iterable[ChatResponse]:
        # Simulate a fake response for testing purposes
        return [
            ChatResponse(
                prompt_token_ids=None,
                num_prompt_tokens=len(input_str),
                prompt_text=input_str,
                output_token_ids=None,
                num_output_tokens=17,
                output_text=f"Token {i}",
                max_token_halt=True,
            )
            for i, input_ in enumerate(inputs)
            if (input_str := cast(str, input_)) is not None
        ]

    def logits(
        self,
        inputs: PromptList,
        **kwargs: Any,
    ) -> Iterable[list[list[TokenPred]]]:
        # Simulate fake logits for testing purposes
        return [
            [[TokenPred(token=f"Token {i}", token_id=i, logprob=-1.0, rank=0)]]
            for i in range(len(inputs))
        ]


@pytest.mark.parametrize(
    "spec, expected_logs",
    [
        (
            BenchmarkSpec(
                name="fake",
                generation_mode=GenerationMode.CHAT_COMPLETION,
                prompt_format=PromptFormatter.BASE,
                n_shot=SampleRatio(0),
            ),
            [
                {
                    "data": {
                        "benchmark_sample_index": 0,
                        "benchmark_sample_hash": "f0",
                        "subject": "apiculture",
                        "system_prompt": None,
                        "instruction": "Answer the following question:",
                        "question": "What is 0?",
                        "choices": None,
                        "label": "0",
                        "formatted_question": "Q: What is 0?",
                        "formatted_answer": "A: 0",
                        "question_prompt": "Answer the following question:\n\nQ: What is 0?",
                    },
                    "model_data": {
                        "answer_symbol_ids": {},
                        "chat_comp": {
                            "prompt_token_ids": None,
                            "num_prompt_tokens": 46,
                            "prompt_text": "Answer the following question:\n\nQ: What is 0?\n",
                            "output_token_ids": None,
                            "num_output_tokens": 17,
                            "output_text": "Fake response to: Answer the following question:\n\nQ: What is 0?\n",
                            "max_token_halt": False,
                        },
                        "prompt": "Answer the following question:\n\nQ: What is 0?\n",
                    },
                    "metadata": ANY,
                },
                {
                    "data": {
                        "benchmark_sample_index": 1,
                        "benchmark_sample_hash": "f1",
                        "subject": "apiculture",
                        "system_prompt": None,
                        "instruction": "Answer the following question:",
                        "question": "What is 1?",
                        "choices": None,
                        "label": "1",
                        "formatted_question": "Q: What is 1?",
                        "formatted_answer": "A: 1",
                        "question_prompt": "Answer the following question:\n\nQ: What is 1?",
                    },
                    "model_data": {
                        "answer_symbol_ids": {},
                        "chat_comp": {
                            "prompt_token_ids": None,
                            "num_prompt_tokens": 46,
                            "prompt_text": "Answer the following question:\n\nQ: What is 1?\n",
                            "output_token_ids": None,
                            "num_output_tokens": 17,
                            "output_text": "Fake response to: Answer the following question:\n\nQ: What is 1?\n",
                            "max_token_halt": False,
                        },
                        "prompt": "Answer the following question:\n\nQ: What is 1?\n",
                    },
                    "metadata": ANY,
                },
            ],
        ),
        (
            BenchmarkSpec(
                name="fake",
                generation_mode=GenerationMode.NEXT_TOKEN,
                prompt_format=PromptFormatter.BASE,
                n_shot=SampleRatio(0),
            ),
            [
                {
                    "data": {
                        "benchmark_sample_index": 0,
                        "benchmark_sample_hash": "f0",
                        "subject": "apiculture",
                        "system_prompt": None,
                        "instruction": "Answer the following question:",
                        "question": "What is 0?",
                        "choices": None,
                        "label": "0",
                        "formatted_question": "Q: What is 0?",
                        "formatted_answer": "A: 0",
                        "question_prompt": "Answer the following question:\n\nQ: What is 0?",
                    },
                    "model_data": {
                        "answer_symbol_ids": {},
                        "next_token": {
                            "max_token_halt": True,
                            "num_output_tokens": 17,
                            "num_prompt_tokens": 49,
                            "output_text": "Token 0",
                            "output_token_ids": None,
                            "prompt_text": "Answer the following question:\n\nQ: What is 0?\nA: ",
                            "prompt_token_ids": None,
                        },
                        "prompt": "Answer the following question:\n\nQ: What is 0?\nA: ",
                    },
                    "metadata": ANY,
                },
                {
                    "data": {
                        "benchmark_sample_index": 1,
                        "benchmark_sample_hash": "f1",
                        "subject": "apiculture",
                        "system_prompt": None,
                        "instruction": "Answer the following question:",
                        "question": "What is 1?",
                        "choices": None,
                        "label": "1",
                        "formatted_question": "Q: What is 1?",
                        "formatted_answer": "A: 1",
                        "question_prompt": "Answer the following question:\n\nQ: What is 1?",
                    },
                    "model_data": {
                        "answer_symbol_ids": {},
                        "next_token": {
                            "max_token_halt": True,
                            "num_output_tokens": 17,
                            "num_prompt_tokens": 49,
                            "output_text": "Token 1",
                            "output_token_ids": None,
                            "prompt_text": "Answer the following question:\n\nQ: What is 1?\nA: ",
                            "prompt_token_ids": None,
                        },
                        "prompt": "Answer the following question:\n\nQ: What is 1?\nA: ",
                    },
                    "metadata": ANY,
                },
            ],
        ),
        (
            BenchmarkSpec(
                name="fake",
                generation_mode=GenerationMode.LOGITS,
                prompt_format=PromptFormatter.BASE,
                n_shot=SampleRatio(0),
            ),
            [
                {
                    "data": {
                        "benchmark_sample_index": 0,
                        "benchmark_sample_hash": "f0",
                        "subject": "apiculture",
                        "system_prompt": None,
                        "instruction": "Answer the following question:",
                        "question": "What is 0?",
                        "choices": None,
                        "label": "0",
                        "formatted_question": "Q: What is 0?",
                        "formatted_answer": "A: 0",
                        "question_prompt": "Answer the following question:\n\nQ: What is 0?",
                    },
                    "model_data": {
                        "answer_symbol_ids": {"A": 87, "B": 31, "C": 7, "D": 9, "E": 5},
                        "logits": [
                            [
                                {
                                    "logprob": -1.0,
                                    "rank": 0,
                                    "token": "Token 0",
                                    "token_id": 0,
                                },
                            ]
                        ],
                        "prompt": "Answer the following question:\n\nQ: What is 0?\nA: ",
                    },
                    "metadata": ANY,
                },
                {
                    "data": {
                        "benchmark_sample_index": 1,
                        "benchmark_sample_hash": "f1",
                        "subject": "apiculture",
                        "system_prompt": None,
                        "instruction": "Answer the following question:",
                        "question": "What is 1?",
                        "choices": None,
                        "label": "1",
                        "formatted_question": "Q: What is 1?",
                        "formatted_answer": "A: 1",
                        "question_prompt": "Answer the following question:\n\nQ: What is 1?",
                    },
                    "model_data": {
                        "answer_symbol_ids": {"A": 87, "B": 31, "C": 7, "D": 9, "E": 5},
                        "logits": [
                            [
                                {
                                    "logprob": -1.0,
                                    "rank": 0,
                                    "token": "Token 1",
                                    "token_id": 1,
                                },
                            ]
                        ],
                        "prompt": "Answer the following question:\n\nQ: What is 1?\nA: ",
                    },
                    "metadata": ANY,
                },
            ],
        ),
    ],
)
def test_query_over_benchmark(
    spec: BenchmarkSpec, expected_logs: dict[str, Any]
) -> None:
    # Define the parameters for the queries.
    gen_params = GenParams(
        temperature=0.5,
        top_p=1.0,
        max_completion_tokens=100,
        kwargs={},
    )

    sampling_params = DataSamplingParams(
        sample_size=None,
    )

    benchmark_logs = list(
        query_over_benchmark(
            FakeBenchmark(
                spec,
                config={
                    "format": {
                        "instructions": {
                            "system": "You are a fake assistant.",
                            "base_inst_template": "Answer the following question:",
                            "chat_inst_template": "Answer the question!",
                        },
                        "prompt": {
                            "question_template": "Q: {{ question }}",
                            "answer_template": "A: {{ answer }}",
                            "prompt_template": "{{ instruction }}\n\n{{ question }}",
                        },
                    },
                },
                seed=147,
            ),
            sampling_params=sampling_params,
            model=FakeModel(model_name="fake-model"),
            gen_params=gen_params,
        )
    )

    assert benchmark_logs == expected_logs
