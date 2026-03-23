# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith._types.benchmark.sample_ratio import SampleRatio


def test_sample_ratio_validation() -> None:
    with pytest.raises(AssertionError, match="Numerator must be non-negative"):
        SampleRatio(-1)
    with pytest.raises(AssertionError, match="Denominator must be positive"):
        SampleRatio(1, 0)
    with pytest.raises(
        AssertionError,
        match="Ratio must be an integer or a fraction with numerator <= denominator",
    ):
        SampleRatio(3, 2)


def test_sample_ratio_str() -> None:
    assert str(SampleRatio(1)) == "1"
    assert str(SampleRatio(1, 1)) == "1"
    assert str(SampleRatio(2, 3)) == "2/3"
    assert str(SampleRatio(0, 5)) == "0/5"
    assert str(SampleRatio(5, 1)) == "5"
    assert str(SampleRatio(5, 5)) == "5/5"


def test_sample_ratio_equality() -> None:
    assert SampleRatio(1) == 1
    assert SampleRatio(1, 2) != 1

    assert SampleRatio(1) == SampleRatio(1, 1)
    assert SampleRatio(2, 3) == SampleRatio(2, 3)
    assert SampleRatio(2, 3) != SampleRatio(2, 4)

    with pytest.raises(TypeError):
        # Ratios are not comparable to floats directly.
        _ = SampleRatio(1, 2) != 0.5  # noqa: B015


def test_sample_ratio_from_string() -> None:
    assert SampleRatio.from_string("1") == SampleRatio(1)
    assert SampleRatio.from_string("2/3") == SampleRatio(2, 3)
    assert SampleRatio.from_string("0") == SampleRatio(0, 1)
    assert SampleRatio.from_string("5") == SampleRatio(5, 1)
    assert SampleRatio.from_string("5/1") == SampleRatio(5, 1)
    assert SampleRatio.from_string("5/5") == SampleRatio(5, 5)
    assert SampleRatio.from_string("5/10") == SampleRatio(5, 10)
