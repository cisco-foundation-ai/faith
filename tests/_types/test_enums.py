# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from enum import auto

import pytest

from faith._types.enums import CIEnum, CIStrEnum


class _Shape(CIEnum):
    CIRCLE = (1,)
    SQUARE = (2,)


def test_ci_enum_name_lookup():
    assert _Shape("circle") is _Shape.CIRCLE
    assert _Shape("CIRCLE") is _Shape.CIRCLE
    assert _Shape("Circle") is _Shape.CIRCLE


def test_ci_enum_str():
    assert str(_Shape.CIRCLE) == "circle"


def test_ci_enum_invalid():
    with pytest.raises(ValueError):
        _Shape("triangle")


class _Color(CIStrEnum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()


def test_ci_str_enum_str_cases():
    assert _Color("red") is _Color.RED
    assert _Color("RED") is _Color.RED
    assert _Color("Red") is _Color.RED


def test_ci_str_enum_str():
    assert str(_Color.RED) == "red"


def test_ci_str_enum_invalid():
    with pytest.raises(ValueError):
        _Color("yellow")
