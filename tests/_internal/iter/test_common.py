# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._internal.iter.common import Functor, GetAttrTransform


class _TestObject:
    def __init__(self, value: int, name: str) -> None:
        self.value = value
        self.name = name


def test_get_attr_transform() -> None:
    """Test the GetAttrTransform to ensure it correctly retrieves attributes."""
    assert not list([] >> GetAttrTransform[_TestObject, int]("value"))
    assert not list([] >> GetAttrTransform[_TestObject, str]("name"))

    objects = [_TestObject(i, f"obj-{i}") for i in range(5)]
    assert list(objects >> GetAttrTransform[_TestObject, int]("value")) == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert list(objects >> GetAttrTransform[_TestObject, str]("name")) == [
        "obj-0",
        "obj-1",
        "obj-2",
        "obj-3",
        "obj-4",
    ]


def test_functor() -> None:
    """Test the Functor to ensure it correctly applies a function to each element."""
    assert not list([] >> Functor[int, int](lambda x: x * 2))

    numbers = [1, 2, 3, 4, 5]
    assert list(numbers >> Functor[int, int](lambda x: x * 2)) == [
        2,
        4,
        6,
        8,
        10,
    ]
