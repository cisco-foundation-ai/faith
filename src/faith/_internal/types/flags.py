# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides types for parsing command line flags."""

from dataclasses import dataclass
from enum import auto

from faith._types.enums import CIStrEnum


class GenerationMode(CIStrEnum):
    """An enumeration of different generation modes for model outputs."""

    LOGITS = auto()
    NEXT_TOKEN = auto()
    CHAT_COMP = auto()


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
