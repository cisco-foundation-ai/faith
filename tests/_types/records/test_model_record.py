# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith._types.records.model_record import ModelRecord
from faith._types.records.model_response import (
    ChatResponse,
    GenerationError,
    TokenPred,
)
from faith.model.params import GenerationMode


@pytest.mark.parametrize("mode", list(GenerationMode))
def test_reset_to_mode_keeps_selected_field(mode: GenerationMode) -> None:
    """reset_to_mode clears all response fields except the one matching the mode."""
    record = ModelRecord(
        prompt="What is 2 + 2?",
        answer_symbol_ids={"A": 0, "B": 1},
        logits=[[TokenPred(token="A", token_id=0, logprob=-1.0)]],
        next_token=ChatResponse(output_text="A"),
        chat_comp=ChatResponse(output_text="The answer is A"),
        error=GenerationError(title="Something went wrong"),
    )
    record.reset_to_mode(mode)

    assert record == ModelRecord(
        prompt="What is 2 + 2?",
        answer_symbol_ids={"A": 0, "B": 1},
        logits=(
            [[TokenPred(token="A", token_id=0, logprob=-1.0)]]
            if mode == GenerationMode.LOGITS
            else None
        ),
        next_token=(
            ChatResponse(output_text="A") if mode == GenerationMode.NEXT_TOKEN else None
        ),
        chat_comp=(
            ChatResponse(output_text="The answer is A")
            if mode == GenerationMode.CHAT_COMP
            else None
        ),
    )
