# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith._internal.validation import assert_same_length


def test_assert_same_length() -> None:
    assert_same_length()
    assert_same_length(a=[], b=[], c=[])
    assert_same_length(a=[1, 2, 3])
    assert_same_length(a=[1, 2, 3], b=[4, 5, 6])
    assert_same_length(expected_length=5)
    assert_same_length(expected_length=3, a=[1, 2, 3], b=[4, 5, 6])

    # Test failure conditions.
    with pytest.raises(
        AssertionError,
        match="All sequences must have the same length. Lengths: {'a': 2, 'b': 1}",
    ):
        assert_same_length(a=[1, 2], b=[3])
    with pytest.raises(
        AssertionError,
        match="All sequences must have the expected length 1. Lengths: {'a': 2, 'b': 2}",
    ):
        assert_same_length(expected_length=1, a=[1, 2], b=[3, 4])
