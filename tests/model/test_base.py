# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import BaseModel, _is_message_list, _is_string_list


def test_is_message_list() -> None:
    assert _is_message_list([[{"role": "user", "content": "Hello"}]])
    assert not _is_message_list(["Hello"])


def test_is_string_list() -> None:
    assert _is_string_list(["Hello", "World"])
    assert not _is_string_list([[{"role": "user", "content": "Hello"}]])


class FakeModel(BaseModel):
    @property
    def supported_formats(self) -> set[PromptFormatter]:
        return set[PromptFormatter]()


def test_base_model_defaults() -> None:
    model = FakeModel("test_model")

    assert model.name_or_path == "test_model"
    assert model.tokenizer is None

    with pytest.raises(NotImplementedError):
        model.logits(inputs=[])
    with pytest.raises(NotImplementedError):
        model.next_token(inputs=[])
    with pytest.raises(NotImplementedError):
        model.query(inputs=[])
