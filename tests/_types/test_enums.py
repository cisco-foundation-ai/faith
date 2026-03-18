# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from enum import auto

import pytest

from faith._types.enums import CIEnum, CIStrEnum


class _Color(CIStrEnum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()


def test_str_cases():
    assert _Color("red") is _Color.RED
    assert _Color("RED") is _Color.RED
    assert _Color("Red") is _Color.RED


def test_str():
    assert str(_Color.RED) == "red"


def test_invalid():
    with pytest.raises(ValueError):
        _Color("yellow")


class _Shape(CIEnum):
    CIRCLE = (1,)
    SQUARE = (2,)


def test_mixin_name_lookup():
    assert _Shape("circle") is _Shape.CIRCLE
    assert _Shape("CIRCLE") is _Shape.CIRCLE
    assert _Shape("Circle") is _Shape.CIRCLE


def test_mixin_str():
    assert str(_Shape.CIRCLE) == "circle"


def test_mixin_invalid():
    with pytest.raises(ValueError):
        _Shape("triangle")
