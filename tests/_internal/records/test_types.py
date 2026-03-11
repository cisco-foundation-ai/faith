# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith._internal.records.types import ModelRecord
from faith._internal.types.flags import GenerationMode


@pytest.mark.parametrize("mode", list(GenerationMode))
def test_reset_to_mode_keeps_selected_field(mode: GenerationMode) -> None:
    """reset_to_mode clears all response fields except the one matching the mode."""
    record = ModelRecord(
        prompt="What is 2 + 2?",
        answer_symbol_ids={"A": 0, "B": 1},
        logits=[[{"token_id": 0, "logprob": -1.0}]],
        next_token={"output_text": "A"},
        chat_comp={"answer_text": "The answer is A"},
        error={"title": "Something went wrong", "details": None},
    )
    record.reset_to_mode(mode)

    assert record == ModelRecord(
        prompt="What is 2 + 2?",
        answer_symbol_ids={"A": 0, "B": 1},
        logits=record.logits if mode == GenerationMode.LOGITS else None,
        next_token=record.next_token if mode == GenerationMode.NEXT_TOKEN else None,
        chat_comp=record.chat_comp if mode == GenerationMode.CHAT_COMPLETION else None,
    )
