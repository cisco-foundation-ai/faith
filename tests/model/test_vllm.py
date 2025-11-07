# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import ChatResponse
from faith.model.vllm import VLLMModel, _remove_longest_common_prefix


@pytest.mark.parametrize(
    "base_list",
    [
        [],
        [7],
        [81, 92],
        [5, 10, 15],
        [1, 2, 3, 4, 5],
        [42, 43, 44, 45, 46, 47, 48],
    ],
)
def test_remove_longest_common_prefix(base_list: list[int]) -> None:
    """Test the _remove_longest_common_prefix function with various cases."""
    for i in range(len(base_list) + 1):
        lst = list(base_list)
        prefix = list(base_list[:i]) + [-7, -8, -9]
        expected = base_list[i:]
        actual = _remove_longest_common_prefix(lst, prefix)
        assert (
            actual == expected
        ), f"Failed for i={i}, lst={lst}, prefix={prefix}: expected {expected}; got {actual}"


@pytest.mark.skip(reason="VLLM requires special setup for CPU testing in C/I.")
def test_vllm_model() -> None:
    """Test the VLLM model initialization and basic functionality."""
    model = VLLMModel(
        name_or_path="EleutherAI/pythia-70m",
        tokenizer_name_or_path=None,
        seed=2651,
        context_len=512,
    )

    assert model.name_or_path == "EleutherAI/pythia-70m"
    assert model.tokenizer is not None
    assert model.supported_formats == {PromptFormatter.BASE}

    # Test next-token generation with the base format.
    assert list(
        model.next_token(
            ["Hello?\n", "What is 1+1?\n"], temperature=0.0, verbose_resps=True
        )
    ) == [
        ChatResponse(
            prompt_token_ids=[12092, 32, 187],
            num_prompt_tokens=3,
            prompt_text="Hello?\n",
            output_token_ids=[29],
            num_output_tokens=1,
            output_text="<",
            request_token_ids=[12092, 32, 187],
            num_request_tokens=3,
            request_text="Hello?\n",
            response_token_ids=[29],
            num_response_tokens=1,
            response_text="<",
            answer_token_ids=[29],
            num_answer_tokens=1,
            answer_text="<",
            max_token_halt=True,
        ),
        ChatResponse(
            prompt_token_ids=[1276, 310, 337, 12, 18, 32, 187],
            num_prompt_tokens=7,
            prompt_text="What is 1+1?\n",
            output_token_ids=[14],
            num_output_tokens=1,
            output_text="-",
            request_token_ids=[1276, 310, 337, 12, 18, 32, 187],
            num_request_tokens=7,
            request_text="What is 1+1?\n",
            response_token_ids=[14],
            num_response_tokens=1,
            response_text="-",
            answer_token_ids=[14],
            num_answer_tokens=1,
            answer_text="-",
            max_token_halt=True,
        ),
    ]

    # Test querying with the base format.
    assert list(
        model.query(
            ["Hello?\n", "What is 1+1?\n"],
            max_completion_tokens=10,
            temperature=0.0,
            verbose_resps=True,
        )
    ) == [
        ChatResponse(
            prompt_token_ids=[12092, 32, 187],
            num_prompt_tokens=3,
            prompt_text="Hello?\n",
            output_token_ids=[29, 37, 757, 31, 309, 1353, 2820, 281, 755, 247],
            num_output_tokens=10,
            output_text="<Dian> I'm trying to get a",
            request_token_ids=[12092, 32, 187],
            num_request_tokens=3,
            request_text="Hello?\n",
            response_token_ids=[29, 37, 757, 31, 309, 1353, 2820, 281, 755, 247],
            num_response_tokens=10,
            response_text="<Dian> I'm trying to get a",
            answer_token_ids=[29, 37, 757, 31, 309, 1353, 2820, 281, 755, 247],
            num_answer_tokens=10,
            answer_text="<Dian> I'm trying to get a",
            max_token_halt=True,
        ),
        ChatResponse(
            prompt_token_ids=[1276, 310, 337, 12, 18, 32, 187],
            num_prompt_tokens=7,
            prompt_text="What is 1+1?\n",
            output_token_ids=[14, 18, 187, 1276, 310, 253, 1318, 273, 428, 18],
            num_output_tokens=10,
            output_text="-1\nWhat is the value of -1",
            request_token_ids=[1276, 310, 337, 12, 18, 32, 187],
            num_request_tokens=7,
            request_text="What is 1+1?\n",
            response_token_ids=[14, 18, 187, 1276, 310, 253, 1318, 273, 428, 18],
            num_response_tokens=10,
            response_text="-1\nWhat is the value of -1",
            answer_token_ids=[14, 18, 187, 1276, 310, 253, 1318, 273, 428, 18],
            num_answer_tokens=10,
            answer_text="-1\nWhat is the value of -1",
            max_token_halt=True,
        ),
    ]

    del model
