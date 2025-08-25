# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from faith._internal.types.flags import (
    AnnotatedPath,
    GenerationMode,
    PathWithAnnotations,
    SampleRatio,
    TypeWithDefault,
)


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


def test_generation_mode_str() -> None:
    assert str(GenerationMode.LOGITS) == "logits"
    assert str(GenerationMode.NEXT_TOKEN) == "next_token"
    assert str(GenerationMode.CHAT_COMPLETION) == "chat_comp"


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
        SampleRatio(1, 2) != 0.5  # noqa: B015


def test_sample_ratio_from_string() -> None:
    assert SampleRatio.from_string("1") == SampleRatio(1)
    assert SampleRatio.from_string("2/3") == SampleRatio(2, 3)
    assert SampleRatio.from_string("0") == SampleRatio(0, 1)
    assert SampleRatio.from_string("5") == SampleRatio(5, 1)
    assert SampleRatio.from_string("5/1") == SampleRatio(5, 1)
    assert SampleRatio.from_string("5/5") == SampleRatio(5, 5)
    assert SampleRatio.from_string("5/10") == SampleRatio(5, 10)


def test_type_with_default() -> None:
    type_with_default = TypeWithDefault(int, 42)
    assert type_with_default("10") == 10
    assert type_with_default(None) == 42

    type_with_default_str = TypeWithDefault(str, "default")
    assert type_with_default_str("hello") == "hello"
    assert type_with_default_str(None) == "default"


def test_path_with_values() -> None:
    path_with_values = PathWithAnnotations("example.txt", version=1.0, format="csv")
    assert path_with_values.path == Path("example.txt")
    assert path_with_values.raw_path == "example.txt"
    assert path_with_values.get_value("version") == 1.0
    assert path_with_values.get_value("format") == "csv"

    with pytest.raises(KeyError):
        path_with_values.get_value("nonexistent_key")


def test_path_with_values_str() -> None:
    path_with_values = PathWithAnnotations("example.txt", version=1.0, format="csv")
    assert str(path_with_values) == "example.txt@version=1.0@format=csv"

    path_with_values_empty = PathWithAnnotations("empty.txt")
    assert str(path_with_values_empty) == "empty.txt"


def test_annotated_path() -> None:
    annotated_path = AnnotatedPath(
        rows=TypeWithDefault(int, 1), format=lambda x: x or "csv"
    )

    pav = annotated_path("path/to/file.json@rows=2@format=json")
    assert pav.path == Path("path/to/file.json")
    assert pav.get_value("rows") == 2
    assert pav.get_value("format") == "json"

    pav = annotated_path("path/to/file.csv@rows=2")
    assert pav.path == Path("path/to/file.csv")
    assert pav.get_value("rows") == 2
    assert pav.get_value("format") == "csv"

    pav = annotated_path("path/to/another_file.txt")
    assert pav.path == Path("path/to/another_file.txt")
    assert pav.get_value("rows") == 1
    assert pav.get_value("format") == "csv"

    with pytest.raises(AssertionError, match="Unknown annotation keys: {'name'}"):
        annotated_path("path/to/file.csv@name=example@rows=2")
