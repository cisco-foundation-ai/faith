# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides types for parsing command line flags."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar


class GenerationMode(Enum):
    """An enumeration of different generation modes for model outputs."""

    LOGITS = "logits"
    NEXT_TOKEN = "next_token"
    CHAT_COMPLETION = "chat_comp"

    def __str__(self) -> str:
        """Return the string representation of the generation mode."""
        return self.value


@dataclass(frozen=True)
class SampleRatio:
    """A class to represent a sample ratio, which can be an integer or a fraction <= 1.

    The ratio is represented as a numerator and a denominator, where the numerator
    is a non-negative integer and the denominator is a positive integer, with the
    numerator <= denominator if the denominator is not 1.

    These ratios are used to specify how many samples to take from a dataset, where
    the denominator represents the total number of samples and the numerator is the
    number of subsamples to use each time a sample is requested. This allows us to
    set aside a fixed population and then re-sample from it multiple times.

    When an integer is provided, this implies each subsample is identical to the
    original sample. For example, a ratio of 5/1 means we select 5 initial samples
    and repeat them each time we request a subsample. In contrast, a ratio of 5/5
    means we select 5 initial samples and then re-sample them for each subsample
    allowing them to be in different orders each time.

    Note: These rations are not simplified, so a ratio of 2/4 is different from 1/2.
    """

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        """Validate the numerator and denominator of the sample ratio."""
        assert self.numerator >= 0, "Numerator must be non-negative"
        assert self.denominator > 0, "Denominator must be positive"
        if self.denominator != 1:
            assert (
                self.numerator <= self.denominator
            ), "Ratio must be an integer or a fraction with numerator <= denominator"

    def __str__(self) -> str:
        """Return the string representation of the sample ratio."""
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"

    def __eq__(self, other: object) -> bool:
        """Check equality with another SampleRatio or an integer."""
        if not isinstance(other, (SampleRatio, int)):
            raise TypeError(f"Cannot compare SampleRatio with {type(other).__name__}")
        if isinstance(other, int):
            return self.numerator == other and self.denominator == 1
        return (
            self.numerator == other.numerator and self.denominator == other.denominator
        )

    @staticmethod
    def from_string(quotient_str: str) -> "SampleRatio":
        """Parse a string in the form of 'numerator/denominator' or 'numerator'."""
        numerator_str, _, denominator_str = quotient_str.partition("/")
        numerator = int(numerator_str)
        denominator = int(denominator_str) if denominator_str else 1
        return SampleRatio(numerator, denominator)


_T = TypeVar("_T")


class TypeWithDefault(Generic[_T]):
    """A type that can be parsed from a string and has a default value."""

    def __init__(self, parse_fn: Callable[[str], _T], default_value: _T) -> None:
        """Initialize the TypeWithDefault with a parsing function and a default value."""
        self._parse_fn = parse_fn
        self._default_value = default_value

    def __call__(self, arg: str | None) -> _T:
        """Parse the `arg` or return the default value if `arg` is None."""
        if arg is None:
            return self._default_value
        return self._parse_fn(arg)


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
            name: parser(overrides.get(name, None))
            for name, parser in self._arg_parsers.items()
        }
        return PathWithAnnotations(raw_path, **parsed_annotations)
