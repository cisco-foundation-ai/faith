# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any, cast

import pytest
from transformers import PreTrainedTokenizerBase

from faith._types.model.generation import GenerationMode, GenParams
from faith._types.model.prompt import PromptFormatter
from faith._types.record.model_response import ChatResponse, GenerationError, TokenPred
from faith._types.record.sample_record import SampleRecord
from faith.model.base import BaseModel, PromptList
from faith.record_pipelines.prediction import model_querier
from tests.benchmark.categories.fake_record_maker import make_fake_record


class FakeModel(BaseModel):
    """A fake model that simulates responses for testing purposes."""

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        """Return the tokenizer for the model."""
        return None

    @property
    def supported_formats(self) -> set[PromptFormatter]:
        """Return the set of prompt formats supported by the fake model."""
        return set(list(PromptFormatter))

    def query(
        self, inputs: PromptList, **_kwargs: Any
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

    def next_token(self, inputs: PromptList, **_kwargs: Any) -> Iterable[ChatResponse]:
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


_PROMPT_NO_LEADIN_0 = "Answer the following question:\n\nQ: What is 0?\n"
_PROMPT_NO_LEADIN_1 = "Answer the following question:\n\nQ: What is 1?\n"
_PROMPT_WITH_LEADIN_0 = "Answer the following question:\n\nQ: What is 0?\nA: "
_PROMPT_WITH_LEADIN_1 = "Answer the following question:\n\nQ: What is 1?\nA: "

_ANSWER_TOKEN_MAP: dict[str, int] = {"A": 87, "B": 31, "C": 7, "D": 9, "E": 5}


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
    model = FakeModel("fake-model")
    transform = model_querier(model, generation_mode, gen_params)

    assert list(transform(input_records)) == expected_records


class ErrorModel(FakeModel):
    """A fake model that always returns GenerationError."""

    def query(
        self, inputs: PromptList, **_kwargs: Any
    ) -> Iterable[ChatResponse | GenerationError]:
        """Simulates a model that always returns a GenerationError."""
        return [GenerationError(title="error") for _ in inputs]

    def next_token(
        self, inputs: PromptList, **_kwargs: Any
    ) -> Iterable[ChatResponse | GenerationError]:
        """Simulates a model that always returns a GenerationError."""
        return [GenerationError(title="error") for _ in inputs]

    def logits(
        self, inputs: PromptList, **_kwargs: Any
    ) -> Iterable[list[list[TokenPred]] | GenerationError]:
        """Simulates a model that always returns a GenerationError."""
        return [GenerationError(title="error") for _ in inputs]


@pytest.mark.parametrize("generation_mode", list(GenerationMode))
def test_model_querier_generation_error(generation_mode: GenerationMode) -> None:
    (result,) = list(
        [make_fake_record(model_data={"prompt": "test", "answer_symbol_ids": {}})]
        >> model_querier(
            ErrorModel("error-model"),
            generation_mode,
            GenParams(temperature=0.0, top_p=1.0, max_completion_tokens=100, kwargs={}),
        )
    )
    assert result.model_data.error == GenerationError(title="error")
