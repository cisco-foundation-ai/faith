# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith._internal.records.sort import SortByTransform
from tests.benchmark.categories.fake_record_maker import make_fake_record


def test_sort_by_nested_field() -> None:
    records = [
        make_fake_record(data={"question": "cherry"}),
        make_fake_record(data={"question": "apple"}),
        make_fake_record(data={"question": "banana"}),
    ]
    result = list(records >> SortByTransform[str]("data", "question"))
    assert [r.data.question for r in result] == ["apple", "banana", "cherry"]


def test_sort_empty() -> None:
    assert not [] >> SortByTransform[str]("data", "question")


def test_sort_invalid_field_path() -> None:
    records = [make_fake_record()]
    with pytest.raises(AttributeError):
        list(records >> SortByTransform("no_such", "field"))
