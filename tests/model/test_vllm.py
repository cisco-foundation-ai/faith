# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import ChatResponse, TokenPred
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


@patch(
    "faith.model.vllm.LLM",
    return_value=Mock(
        get_tokenizer=Mock(return_value=Mock(vocab_size=50000, chat_template=None)),
    ),
)
def test_vllm_model_init_without_reasoning_tokens(_mock_llm_class: Mock) -> None:
    """Test VLLMModel initialization without reasoning tokens."""
    # Create the VLLMModel around the mocked LLM class.
    model = VLLMModel(
        name_or_path="test-model",
        tokenizer_name_or_path="test-tokenizer",
        num_gpus=1,
        seed=54748,
        context_len=400,
    )

    # Test that the model has been initialized with the correct reasoning tokens.
    assert model.name_or_path == "test-model"
    assert model._reasoning_tokens is None  # pylint: disable=protected-access
    assert model.supported_formats == {PromptFormatter.BASE}


@patch(
    "faith.model.vllm.LLM",
    return_value=Mock(
        get_tokenizer=Mock(
            return_value=Mock(
                vocab_size=50000,
                chat_template="some_template",
                encode=Mock(side_effect=lambda x: [100] if x == "<think>" else [101]),
            )
        ),
    ),
)
def test_vllm_model_init_with_reasoning_tokens_as_strings(
    _mock_llm_class: Mock,
) -> None:
    """Test VLLMModel initialization with reasoning tokens as strings."""
    # Create the VLLMModel around the mocked LLM class.
    model = VLLMModel(
        name_or_path="test-model",
        tokenizer_name_or_path="test-tokenizer",
        reasoning_tokens=("<think>", "</think>"),
    )

    # Test that the model has been initialized with the correct reasoning tokens.
    assert model.name_or_path == "test-model"
    assert model._reasoning_tokens == ([100], [101])  # pylint: disable=protected-access
    assert model.supported_formats == {PromptFormatter.BASE, PromptFormatter.CHAT}


@patch(
    "faith.model.vllm.LLM",
    return_value=Mock(
        get_tokenizer=Mock(return_value=Mock(vocab_size=50000, chat_template=None)),
    ),
)
def test_vllm_model_init_with_reasoning_tokens_as_ids(_mock_llm_class: Mock) -> None:
    """Test VLLMModel initialization with reasoning tokens as numeric IDs."""
    # Create the VLLMModel around the mocked LLM class.
    model = VLLMModel(
        name_or_path="test-model",
        tokenizer_name_or_path="test-tokenizer",
        reasoning_tokens=([100], [101]),
    )

    # Test that the model has been initialized with the correct reasoning tokens.
    assert model.name_or_path == "test-model"
    assert model._reasoning_tokens == ([100], [101])  # pylint: disable=protected-access
    assert model.supported_formats == {PromptFormatter.BASE}


# Patch the LLM class to return a mock LLM instance that mocks the get_tokenizer
# and generate methods to simulate the behavior of a real LLM with logprobs.
@patch(
    "faith.model.vllm.LLM",
    return_value=Mock(
        get_tokenizer=Mock(
            return_value=Mock(
                vocab_size=50000,
                decode=Mock(side_effect=lambda x, **kwargs: f"token_{x}"),
            )
        ),
        generate=Mock(
            # Create mock output with logprobs.
            return_value=[
                Mock(
                    outputs=[
                        Mock(
                            logprobs=[
                                {
                                    100: Mock(logprob=-0.5, rank=1),
                                    101: Mock(logprob=-1.2, rank=2),
                                }
                            ],
                        )
                    ],
                )
            ]
        ),
    ),
)
def test_vllm_model_logits(_mock_llm_class: Mock) -> None:
    """Test VLLMModel logits method returns token predictions with log probabilities."""
    # Create the VLLMModel around the mocked LLM class.
    model = VLLMModel(
        name_or_path="test-model",
        tokenizer_name_or_path="test-tokenizer",
        num_log_probs=1,
    )

    # Test logits and check resulting TokenPred objects derived from the mock outputs.
    assert list(model.logits(["Test input"], max_answer_tokens=1)) == [
        [
            [
                TokenPred(token="token_100", token_id=100, logprob=-0.5, rank=1),
                TokenPred(token="token_101", token_id=101, logprob=-1.2, rank=2),
            ],
        ],
    ]


# Patch the LLM class to return a mock LLM instance that mocks the get_tokenizer
# and generate methods to simulate the behavior of a real LLM.
@patch(
    "faith.model.vllm.LLM",
    return_value=Mock(
        get_tokenizer=Mock(
            return_value=Mock(
                vocab_size=50000,
                encode=Mock(return_value=[1, 2, 3]),
                decode=Mock(side_effect=lambda x, **kwargs: "decoded_output"),
            )
        ),
        generate=Mock(
            return_value=[
                Mock(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        Mock(
                            token_ids=[4],
                            text="next",
                            finish_reason="length",
                            logprobs=None,
                        )
                    ],
                )
            ]
        ),
    ),
)
def test_vllm_model_next_token(_mock_llm_class: Mock) -> None:
    """Test that VLLMModel's next_token method generates single token."""
    # Create the VLLMModel around the mocked LLM class.
    model = VLLMModel(
        name_or_path="test-model", tokenizer_name_or_path="test-tokenizer"
    )

    # Test query and check resulting ChatResponse object derived from the mock outputs.
    assert list(model.next_token(["Test input"], temperature=0.0)) == [
        ChatResponse(
            prompt_token_ids=None,
            num_prompt_tokens=3,
            prompt_text=None,
            output_token_ids=None,
            num_output_tokens=1,
            output_text="next",
            request_token_ids=None,
            num_request_tokens=3,
            request_text=None,
            response_token_ids=None,
            num_response_tokens=1,
            response_text=None,
            answer_token_ids=None,
            num_answer_tokens=1,
            answer_text="decoded_output",
            max_token_halt=True,
        )
    ]


# Patch the LLM class to return a mock LLM instance that mocks the get_tokenizer
# and generate methods to simulate the behavior of a real LLM.
@patch(
    "faith.model.vllm.LLM",
    return_value=Mock(
        get_tokenizer=Mock(
            return_value=Mock(
                vocab_size=50000,
                encode=Mock(return_value=[1, 2, 3]),
                decode=Mock(
                    side_effect=lambda x, **kwargs: (
                        "input_text" if x[0] != 4 else "output_text"
                    )
                ),
            )
        ),
        generate=Mock(
            return_value=[
                Mock(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        Mock(
                            token_ids=[4, 5, 6, 7, 8],
                            text="This is a full response",
                            finish_reason="stop",
                            logprobs=None,
                        )
                    ],
                )
            ]
        ),
    ),
)
def test_vllm_model_query(_mock_llm_class: Mock) -> None:
    """Test that VLLMModel's query method generates full response."""

    # Create the VLLMModel around the mocked LLM class.
    model = VLLMModel(
        name_or_path="test-model", tokenizer_name_or_path="test-tokenizer"
    )

    # Test query and check resulting ChatResponse object derived from the mock outputs.
    assert list(model.query(["Test input"], max_completion_tokens=10)) == [
        ChatResponse(
            prompt_token_ids=None,
            num_prompt_tokens=3,
            prompt_text=None,
            output_token_ids=None,
            num_output_tokens=5,
            output_text="This is a full response",
            request_token_ids=None,
            num_request_tokens=3,
            request_text=None,
            response_token_ids=None,
            num_response_tokens=5,
            response_text=None,
            answer_token_ids=None,
            num_answer_tokens=5,
            answer_text="output_text",
            max_token_halt=False,
        )
    ]
    assert list(
        model.query(["Test input"], verbose_resps=True, max_completion_tokens=10)
    ) == [
        ChatResponse(
            prompt_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
            prompt_text="input_text",
            output_token_ids=[4, 5, 6, 7, 8],
            num_output_tokens=5,
            output_text="This is a full response",
            request_token_ids=[1, 2, 3],
            num_request_tokens=3,
            request_text="input_text",
            response_token_ids=[4, 5, 6, 7, 8],
            num_response_tokens=5,
            response_text="output_text",
            answer_token_ids=[4, 5, 6, 7, 8],
            num_answer_tokens=5,
            answer_text="output_text",
            max_token_halt=False,
        )
    ]


# Patch the LLM class to return a mock LLM instance that mocks the get_tokenizer
# and chat methods to simulate the behavior of a real LLM.
# The outputs of the chat method are designed to test reasoning token scenarios.
@patch(
    "faith.model.vllm.LLM",
    return_value=Mock(
        # Use a mock tokenizer to the LLM instance that can encode/decode tokens.
        get_tokenizer=Mock(
            return_value=Mock(
                vocab_size=50000,
                chat_template="some_template",
                apply_chat_template=Mock(return_value=[1, 2, 3]),
                decode=Mock(
                    side_effect=lambda x, **kwargs: (
                        "answer" if x == [8, 9] else f"decoded_{len(x)}"
                    )
                ),
            )
        ),
        chat=Mock(
            return_value=[
                # A mock output with no reasoning tokens.
                # Response: [answer]
                Mock(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        Mock(
                            token_ids=[8, 9],
                            text="answer",
                            finish_reason="stop",
                            logprobs=None,
                        )
                    ],
                ),
                # A mock output with reasoning tokens.
                # Response: [<think>, reasoning, </think>, answer]
                Mock(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        Mock(
                            token_ids=[100, 4, 5, 101, 7, 8, 9],
                            text="<think>reasoning</think>answer",
                            finish_reason="stop",
                            logprobs=None,
                        )
                    ],
                ),
                # A mock output with incomplete reasoning tokens.
                # Response: [<think>, reasoning, </think>, answer, <think>, reasoning2]
                Mock(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[
                        Mock(
                            token_ids=[4, 100, 4, 5, 101, 7, 8, 9, 100, 10, 11, 101],
                            text="reason<think>reasoning</think>answer<think>more</",
                            finish_reason="length",
                            logprobs=None,
                        )
                    ],
                ),
            ]
        ),
    ),
)
def test_vllm_model_query_with_reasoning_tokens(_mock_llm_class: Mock) -> None:
    """Test that VLLMModel's query extracts answer tokens when reasoning tokens are configured."""
    # Create the VLLMModel with reasoning tokens configured.
    model = VLLMModel(
        name_or_path="test-model",
        tokenizer_name_or_path="test-tokenizer",
        reasoning_tokens=([100], [101, 7]),
    )

    # Test query and check resulting ChatResponse object derived from the mock outputs.
    assert list(
        model.query(
            [
                [{"role": "user", "content": "Knock, knock"}],
                [{"role": "user", "content": "Hello"}],
                [{"role": "user", "content": "Who's there?"}],
            ],
            max_completion_tokens=10,
            verbose_resps=True,
        )
    ) == [
        ChatResponse(
            prompt_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
            prompt_text="decoded_3",
            output_token_ids=[8, 9],
            num_output_tokens=2,
            output_text="answer",
            request_token_ids=[1, 2, 3],
            num_request_tokens=3,
            request_text="decoded_3",
            response_token_ids=[8, 9],
            num_response_tokens=2,
            response_text="answer",
            answer_token_ids=[8, 9],
            num_answer_tokens=2,
            answer_text="answer",
            max_token_halt=False,
        ),
        ChatResponse(
            prompt_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
            prompt_text="decoded_3",
            output_token_ids=[100, 4, 5, 101, 7, 8, 9],
            num_output_tokens=7,
            output_text="<think>reasoning</think>answer",
            request_token_ids=[1, 2, 3],
            num_request_tokens=3,
            request_text="decoded_3",
            response_token_ids=[100, 4, 5, 101, 7, 8, 9],
            num_response_tokens=7,
            response_text="decoded_7",
            answer_token_ids=[8, 9],
            num_answer_tokens=2,
            answer_text="answer",
            max_token_halt=False,
        ),
        ChatResponse(
            prompt_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
            prompt_text="decoded_3",
            output_token_ids=[4, 100, 4, 5, 101, 7, 8, 9, 100, 10, 11, 101],
            num_output_tokens=12,
            output_text="reason<think>reasoning</think>answer<think>more</",
            request_token_ids=[1, 2, 3],
            num_request_tokens=3,
            request_text="decoded_3",
            response_token_ids=[4, 100, 4, 5, 101, 7, 8, 9, 100, 10, 11, 101],
            num_response_tokens=12,
            response_text="decoded_12",
            answer_token_ids=[],
            num_answer_tokens=0,
            answer_text="decoded_0",
            max_token_halt=True,
        ),
    ]


@pytest.mark.skip(reason="VLLM requires special setup for CPU testing in C/I.")
def test_vllm_model_with_real_model() -> None:
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
