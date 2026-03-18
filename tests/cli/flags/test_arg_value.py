# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith.cli.flags.arg_value import (
    DefaultValue,
    TypeWithDefault,
    UserValue,
    UserValueType,
)


def test_default_value() -> None:
    d = DefaultValue[float](0.0)
    assert d.value == 0.0
    assert d.is_default is True
    assert repr(d) == "DefaultValue(0.0)"


def test_user_value() -> None:
    u = UserValue[float](3.14)
    assert u.value == 3.14
    assert u.is_default is False
    assert repr(u) == "UserValue(3.14)"


def test_arg_value_equality() -> None:
    assert DefaultValue[int](1) == DefaultValue[int](1)
    assert DefaultValue[int](1) == UserValue[int](1)
    assert DefaultValue[int](1) != DefaultValue[int](2)
    assert DefaultValue[int](1) != "not an ArgValue"


def test_arg_value_hash() -> None:
    assert hash(DefaultValue[int](1)) == hash(DefaultValue[int](1))
    assert hash(DefaultValue[int](1)) != hash(DefaultValue[int](2))
    assert hash(UserValue[int](5)) == hash(UserValue[int](5))
    assert hash(UserValue[str]("5")) != hash(UserValue[int](5))
    assert hash(DefaultValue[int](1)) == hash(UserValue[int](1))


def test_user_value_type() -> None:
    assert UserValueType(int)("10").value == 10
    assert not UserValueType(int)("10").is_default
    assert UserValueType(float)("3.14").value == 3.14
    assert UserValueType(str)("hello").value == "hello"


def test_type_with_default() -> None:
    type_with_default = TypeWithDefault(int, 42)
    assert type_with_default("10") == 10
    assert type_with_default(None) == 42

    type_with_default_str = TypeWithDefault(str, "default")
    assert type_with_default_str("hello") == "hello"
    assert type_with_default_str(None) == "default"
