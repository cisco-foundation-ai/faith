# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import ANY

import numpy as np
import pandas as pd
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from faith._internal.algo.sampling import NShotSampler
from faith._internal.io.json import write_as_json
from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.benchmark.spec import BenchmarkSpec
from faith._types.config.benchmark import BenchmarkConfig
from faith._types.config.format import FormatConfig, InstructionsConfig, PromptConfig
from faith._types.config.patterns import PatternDef
from faith._types.model.generation import GenerationMode, GenParams
from faith._types.model.prompt import PromptFormatter
from faith._types.record.model_response import ChatResponse, GenerationError, TokenPred
from faith._types.record.prompt_record import PromptRecord
from faith._types.record.sample_record import SampleRecord
from faith.benchmark.benchmark import Benchmark, BenchmarkDataset
from faith.benchmark.formatting.qa import QAFormatter
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader
from faith.cli.subcmd.query import BenchmarkRecordTransform, model_querier
from faith.model.base import BaseModel, PromptList
from tests.benchmark.categories.fake_record_maker import make_fake_record


class FakeBenchmark(Benchmark):
    """A fake benchmark for testing purposes."""

    def answer_leadin(self, _tokenizer: PreTrainedTokenizerBase) -> str:
        """Return the lead-in string for the answer."""
        return "A: "

    def answer_token_map(self, _tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
        """Return a mapping of answer tokens to their IDs."""
        return {"A": 87, "B": 31, "C": 7, "D": 9, "E": 5}

    def build_dataset(
        self,
        _sample_size: int | None = None,
        _randomize_choices: bool = False,
    ) -> BenchmarkDataset:
        """Return the dataset for the fake benchmark."""
        return FakeDataset(self.formatter, np.random.default_rng())

    def log_grader(
        self,
        *,
        model_format_config: PatternDef | None = None,
        recompute_stats: bool = False,
    ) -> LogGrader:
        """Return the log grader for the benchmark."""
        raise NotImplementedError("This method should not be called.")

    def grade_aggregator(self) -> GradeAggregator:
        """Return the grade aggregator for the benchmark."""
        raise NotImplementedError("This method should not be called.")


class FakeDataset(BenchmarkDataset):
    """A fake dataset for testing purposes."""

    def __init__(self, formatter: QAFormatter, rng: np.random.Generator):
        super().__init__(
            formatter,
            # Generate fake data for testing purposes
            pd.DataFrame(
                {
                    "question": ["What is 0?", "What is 1?"],
                    "answer": ["0", "1"],
                    "other_data": ["foo", "bar"],
                }
            ),
            NShotSampler(None, SampleRatio(0), rng),
            rng,
            required_columns=frozenset({"question", "answer"}),
            ancillary_columns=frozenset({"other_data"}),
        )

    def _format_qa(
        self,
        index: int,
        sample: pd.Series,
        examples: Sequence[PromptRecord] | None = None,
    ) -> PromptRecord:
        return self._formatter.render_qa_record(
            index=index,
            sample_hash=f"f{index}",
            raw_question=sample["question"],
            raw_answer=sample["answer"],
            examples=examples,
            subject="apiculture",
            ancillary_data=self._extract_ancillary_data(sample),
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
        """Return the set of prompt formats supported by the fake model."""
        return set(list(PromptFormatter))

    def query(
        self,
        inputs: PromptList,
        **_kwargs: Any,
    ) -> Iterable[ChatResponse | GenerationError]:
        """Simulates a fake response for testing purposes."""
        return [
            ChatResponse(
                output_text=f"Fake response to: {input_str}",
                num_output_tokens=17,
                prompt_text=input_str,
                num_prompt_tokens=len(input_str),
                request_text=input_str[:-1],
                num_request_tokens=len(input_str) - 1,
                response_text=input_str[-1] + f"Fake response to: {input_str}",
                num_response_tokens=18,
                answer_text=input_str[-1] + f"Fake response to: {input_str}",
                num_answer_tokens=18,
                max_token_halt=False,
            )
            for input_ in inputs
            if (input_str := cast(str, input_)) is not None
        ]

    def next_token(
        self,
        inputs: PromptList,
        **_kwargs: Any,
    ) -> Iterable[ChatResponse]:
        """Simulates a fake response for testing purposes."""
        return [
            ChatResponse(
                output_text=f"Token {i}",
                num_output_tokens=17,
                prompt_text=input_str,
                num_prompt_tokens=len(input_str),
                request_text=input_str[:-1],
                num_request_tokens=len(input_str) - 1,
                response_text=input_str[-1] + f"Token {i}",
                num_response_tokens=18,
                answer_text=input_str[-1] + f"Token {i}",
                num_answer_tokens=18,
                max_token_halt=True,
            )
            for i, input_ in enumerate(inputs)
            if (input_str := cast(str, input_)) is not None
        ]

    def logits(
        self,
        inputs: PromptList,
        **_kwargs: Any,
    ) -> Iterable[list[list[TokenPred]]]:
        """Simulates fake logits for testing purposes."""
        return [
            [[TokenPred(token=f"Token {i}", token_id=i, logprob=-1.0, rank=0)]]
            for i in range(len(inputs))
        ]


_FAKE_BENCHMARK_CONFIG = BenchmarkConfig(
    format=FormatConfig(
        instructions=InstructionsConfig(
            system_prompt_template="You are a fake assistant.",
            base_inst_template="Answer the following question:",
            chat_inst_template="Answer the question!",
        ),
        prompt=PromptConfig(
            question_template="Q: {{ question }}",
            answer_template="A: {{ answer }}",
            prompt_template="{{ instruction }}\n\n{{ question }}",
        ),
    )
)

_DATA_RECORD_0: dict[str, Any] = {
    "benchmark_sample_index": 0,
    "benchmark_sample_hash": "f0",
    "subject": "apiculture",
    "system_prompt": "You are a fake assistant.",
    "instruction": "Answer the following question:",
    "question": "What is 0?",
    "choices": None,
    "label": "0",
    "formatted_question": "Q: What is 0?",
    "formatted_answer": "A: 0",
    "question_prompt": "Answer the following question:\n\nQ: What is 0?",
    "ancillary_data": {"other_data": "foo"},
}

_DATA_RECORD_1: dict[str, Any] = {
    "benchmark_sample_index": 1,
    "benchmark_sample_hash": "f1",
    "subject": "apiculture",
    "system_prompt": "You are a fake assistant.",
    "instruction": "Answer the following question:",
    "question": "What is 1?",
    "choices": None,
    "label": "1",
    "formatted_question": "Q: What is 1?",
    "formatted_answer": "A: 1",
    "question_prompt": "Answer the following question:\n\nQ: What is 1?",
    "ancillary_data": {"other_data": "bar"},
}

_PROMPT_NO_LEADIN_0 = "Answer the following question:\n\nQ: What is 0?\n"
_PROMPT_NO_LEADIN_1 = "Answer the following question:\n\nQ: What is 1?\n"
_PROMPT_WITH_LEADIN_0 = "Answer the following question:\n\nQ: What is 0?\nA: "
_PROMPT_WITH_LEADIN_1 = "Answer the following question:\n\nQ: What is 1?\nA: "

_ANSWER_TOKEN_MAP: dict[str, int] = {"A": 87, "B": 31, "C": 7, "D": 9, "E": 5}


@pytest.mark.parametrize(
    "spec, expected_records",
    [
        (
            BenchmarkSpec(
                name="fake",
                generation_mode=GenerationMode.CHAT_COMP,
                prompt_format=PromptFormatter.BASE,
                n_shot=SampleRatio(0),
            ),
            [
                {
                    "data": _DATA_RECORD_0,
                    "model_data": {
                        "answer_symbol_ids": {},
                        "prompt": _PROMPT_NO_LEADIN_0,
                    },
                    "metadata": ANY,
                },
                {
                    "data": _DATA_RECORD_1,
                    "model_data": {
                        "answer_symbol_ids": {},
                        "prompt": _PROMPT_NO_LEADIN_1,
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
                    "data": _DATA_RECORD_0,
                    "model_data": {
                        "answer_symbol_ids": {},
                        "prompt": _PROMPT_WITH_LEADIN_0,
                    },
                    "metadata": ANY,
                },
                {
                    "data": _DATA_RECORD_1,
                    "model_data": {
                        "answer_symbol_ids": {},
                        "prompt": _PROMPT_WITH_LEADIN_1,
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
                    "data": _DATA_RECORD_0,
                    "model_data": {
                        "answer_symbol_ids": _ANSWER_TOKEN_MAP,
                        "prompt": _PROMPT_WITH_LEADIN_0,
                    },
                    "metadata": ANY,
                },
                {
                    "data": _DATA_RECORD_1,
                    "model_data": {
                        "answer_symbol_ids": _ANSWER_TOKEN_MAP,
                        "prompt": _PROMPT_WITH_LEADIN_1,
                    },
                    "metadata": ANY,
                },
            ],
        ),
    ],
)
def test_benchmark_record_transform(
    spec: BenchmarkSpec, expected_records: list[dict[str, Any]]
) -> None:
    model = FakeModel(model_name="fake-model")
    benchmark = FakeBenchmark(spec, config=_FAKE_BENCHMARK_CONFIG, seed=147)

    assert [
        rec.to_dict()
        for rec in benchmark.build_dataset().iter_data()
        >> BenchmarkRecordTransform(benchmark, model.tokenizer)
    ] == expected_records


@pytest.mark.parametrize(
    "generation_mode, gen_params, input_records, expected_records",
    [
        (
            GenerationMode.CHAT_COMP,
            GenParams(temperature=0.5, top_p=1.0, max_completion_tokens=100, kwargs={}),
            [
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_NO_LEADIN_0,
                        "answer_symbol_ids": {},
                    }
                ),
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_NO_LEADIN_1,
                        "answer_symbol_ids": {},
                    }
                ),
            ],
            [
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_NO_LEADIN_0,
                        "answer_symbol_ids": {},
                        "chat_comp": {
                            "prompt_token_ids": None,
                            "num_prompt_tokens": 46,
                            "prompt_text": _PROMPT_NO_LEADIN_0,
                            "output_token_ids": None,
                            "num_output_tokens": 17,
                            "output_text": f"Fake response to: {_PROMPT_NO_LEADIN_0}",
                            "max_token_halt": False,
                            "request_token_ids": None,
                            "num_request_tokens": 45,
                            "request_text": "Answer the following question:\n\nQ: What is 0?",
                            "response_token_ids": None,
                            "num_response_tokens": 18,
                            "response_text": f"\nFake response to: {_PROMPT_NO_LEADIN_0}",
                            "answer_token_ids": None,
                            "num_answer_tokens": 18,
                            "answer_text": f"\nFake response to: {_PROMPT_NO_LEADIN_0}",
                        },
                    },
                ),
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_NO_LEADIN_1,
                        "answer_symbol_ids": {},
                        "chat_comp": {
                            "prompt_token_ids": None,
                            "num_prompt_tokens": 46,
                            "prompt_text": _PROMPT_NO_LEADIN_1,
                            "output_token_ids": None,
                            "num_output_tokens": 17,
                            "output_text": f"Fake response to: {_PROMPT_NO_LEADIN_1}",
                            "max_token_halt": False,
                            "request_token_ids": None,
                            "num_request_tokens": 45,
                            "request_text": "Answer the following question:\n\nQ: What is 1?",
                            "response_token_ids": None,
                            "num_response_tokens": 18,
                            "response_text": f"\nFake response to: {_PROMPT_NO_LEADIN_1}",
                            "answer_token_ids": None,
                            "num_answer_tokens": 18,
                            "answer_text": f"\nFake response to: {_PROMPT_NO_LEADIN_1}",
                        },
                    },
                ),
            ],
        ),
        (
            GenerationMode.NEXT_TOKEN,
            GenParams(temperature=0.5, top_p=1.0, max_completion_tokens=100, kwargs={}),
            [
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_WITH_LEADIN_0,
                        "answer_symbol_ids": {},
                    }
                ),
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_WITH_LEADIN_1,
                        "answer_symbol_ids": {},
                    }
                ),
            ],
            [
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_WITH_LEADIN_0,
                        "answer_symbol_ids": {},
                        "next_token": {
                            "prompt_token_ids": None,
                            "num_prompt_tokens": 49,
                            "prompt_text": _PROMPT_WITH_LEADIN_0,
                            "output_token_ids": None,
                            "num_output_tokens": 17,
                            "output_text": "Token 0",
                            "max_token_halt": True,
                            "request_token_ids": None,
                            "num_request_tokens": 48,
                            "request_text": "Answer the following question:\n\nQ: What is 0?\nA:",
                            "response_token_ids": None,
                            "num_response_tokens": 18,
                            "response_text": " Token 0",
                            "answer_token_ids": None,
                            "num_answer_tokens": 18,
                            "answer_text": " Token 0",
                        },
                    },
                ),
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_WITH_LEADIN_1,
                        "answer_symbol_ids": {},
                        "next_token": {
                            "prompt_token_ids": None,
                            "num_prompt_tokens": 49,
                            "prompt_text": _PROMPT_WITH_LEADIN_1,
                            "output_token_ids": None,
                            "num_output_tokens": 17,
                            "output_text": "Token 1",
                            "max_token_halt": True,
                            "request_token_ids": None,
                            "num_request_tokens": 48,
                            "request_text": "Answer the following question:\n\nQ: What is 1?\nA:",
                            "response_token_ids": None,
                            "num_response_tokens": 18,
                            "response_text": " Token 1",
                            "answer_token_ids": None,
                            "num_answer_tokens": 18,
                            "answer_text": " Token 1",
                        },
                    },
                ),
            ],
        ),
        (
            GenerationMode.LOGITS,
            GenParams(temperature=0.5, top_p=1.0, max_completion_tokens=100, kwargs={}),
            [
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_WITH_LEADIN_0,
                        "answer_symbol_ids": _ANSWER_TOKEN_MAP,
                    }
                ),
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_WITH_LEADIN_1,
                        "answer_symbol_ids": _ANSWER_TOKEN_MAP,
                    }
                ),
            ],
            [
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_WITH_LEADIN_0,
                        "answer_symbol_ids": _ANSWER_TOKEN_MAP,
                        "logits": [
                            [
                                {
                                    "token": "Token 0",
                                    "token_id": 0,
                                    "logprob": -1.0,
                                    "rank": 0,
                                },
                            ]
                        ],
                    },
                ),
                make_fake_record(
                    model_data={
                        "prompt": _PROMPT_WITH_LEADIN_1,
                        "answer_symbol_ids": _ANSWER_TOKEN_MAP,
                        "logits": [
                            [
                                {
                                    "token": "Token 1",
                                    "token_id": 1,
                                    "logprob": -1.0,
                                    "rank": 0,
                                },
                            ]
                        ],
                    },
                ),
            ],
        ),
    ],
)
def test_model_querier(
    generation_mode: GenerationMode,
    gen_params: GenParams,
    input_records: list[SampleRecord],
    expected_records: list[SampleRecord],
) -> None:
    model = FakeModel(model_name="fake-model")
    transform = model_querier(model, generation_mode, gen_params)

    assert list(transform(input_records)) == expected_records
