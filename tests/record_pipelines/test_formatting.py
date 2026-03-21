# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
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
from faith._types.model.generation import GenerationMode
from faith._types.model.prompt import PromptFormatter
from faith._types.record.prompt import PromptRecord
from faith.benchmark.benchmark import Benchmark, BenchmarkDataset
from faith.benchmark.formatting.qa import QAFormatter
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader
from faith.record_pipelines.formatting import SampleFormatter


def _make_fake_tokenizer() -> PreTrainedTokenizerBase:
    """Create a minimal fake tokenizer for testing."""
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
        return AutoTokenizer.from_pretrained(tmp_path)


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
def test_sample_formatter(
    spec: BenchmarkSpec, expected_records: list[dict[str, Any]]
) -> None:
    benchmark = FakeBenchmark(spec, config=_FAKE_BENCHMARK_CONFIG, seed=147)

    assert [
        rec.to_dict()
        for rec in benchmark.build_dataset().iter_data()
        >> SampleFormatter(benchmark, _make_fake_tokenizer())
    ] == expected_records
