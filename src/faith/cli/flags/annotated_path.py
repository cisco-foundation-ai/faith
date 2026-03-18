# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides types for parsing paths with inline annotations."""

from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable


class PathWithAnnotations:
    """A class to hold a path and its annotated values."""

    def __init__(self, raw_path: str, **values: Any) -> None:
        """Initialize the PathWithAnnotations with a path and its annotated values."""
        self._raw_path = raw_path
        self._path = Path(raw_path)
        self._values = values

    def __str__(self) -> str:
        """Return a string representation of the path with its annotations."""
        annotations = "@".join(
            f"{key}={str(value)}" for key, value in self._values.items()
        )
        return f"{self._path}{'@' + annotations if annotations else ''}"

    @property
    def path(self) -> Path:
        """Return the path from which to load the underlying object."""
        return self._path

    @property
    def raw_path(self) -> str:
        """Return the raw string representation of the path."""
        return self._raw_path

    def get_value(self, key: str) -> Any:
        """Get the value associated with a specific annotation key."""
        return self._values[key]

    def values(self) -> MappingProxyType[str, Any]:
        """Return a dictionary of all annotation keys and their associated values."""
        return MappingProxyType(self._values)


class AnnotatedPath:
    """A class to parse a path with annotations in the form of `path@key=value@...`."""

    def __init__(self, **arg_parsers: Callable[[str | None], Any]) -> None:
        """Initialize the AnnotatedPath with parsers for each annotation key."""
        self._arg_parsers: dict[str, Callable[[str | None], Any]] = arg_parsers

    def __call__(self, arg: str) -> PathWithAnnotations:
        """Parse a string with annotations in the form of `path@key=value@...`."""
        raw_path, *annotations = arg.split("@")
        assert all(
            "=" in a for a in annotations
        ), f"Annotations must be in the form of key=value, got: {annotations}"
        overrides = {
            name: value
            for a in annotations
            for name, _, value in [a.partition("=")]
            if value
        }
        assert set(overrides.keys()).issubset(
            self._arg_parsers.keys()
        ), f"Unknown annotation keys: {set(overrides.keys()) - set(self._arg_parsers.keys())}"
        parsed_annotations = {
            name: parser(overrides.get(name))
            for name, parser in self._arg_parsers.items()
        }
        return PathWithAnnotations(raw_path, **parsed_annotations)
