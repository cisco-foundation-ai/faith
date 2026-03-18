# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from faith.cli.flags.annotated_path import AnnotatedPath, PathWithAnnotations
from faith.cli.flags.arg_value import TypeWithDefault


def test_path_with_values() -> None:
    path_with_values = PathWithAnnotations("example.txt", version=1.0, format="csv")
    assert path_with_values.path == Path("example.txt")
    assert path_with_values.raw_path == "example.txt"
    assert path_with_values.get_value("version") == 1.0
    assert path_with_values.get_value("format") == "csv"
    assert path_with_values.values() == {"version": 1.0, "format": "csv"}

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
