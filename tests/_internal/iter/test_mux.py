# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Iterable

from faith._internal.iter.mux import MuxTransform
from faith._internal.iter.transform import IdentityTransform, IsoTransform


class SquareTransform(IsoTransform[int]):
    """A simple transform that squares each integer in the generator."""

    def __call__(self, it: Iterable[int]) -> Iterable[int]:
        """Yield the square of each integer from the input generator."""
        for item in it:
            yield item**2


class CubeTransform(IsoTransform[int]):
    """A simple transform that cubes each integer in the generator."""

    def __call__(self, it: Iterable[int]) -> Iterable[int]:
        """Yield the cube of each integer from the input generator."""
        for item in it:
            yield item**3


class Mod7Type(Enum):
    TwoFour = 0
    ThreeFive = 1
    Other = 2

    @staticmethod
    def from_int(value: int) -> "Mod7Type":
        if value % 7 in [2, 4]:
            return Mod7Type.TwoFour
        if value % 7 in [3, 5]:
            return Mod7Type.ThreeFive
        return Mod7Type.Other


def test_mux_transform() -> None:
    """Test the MuxTransform with different kinds of transforms."""
    powers_mux = MuxTransform[Mod7Type, int, int](
        {
            Mod7Type.TwoFour: SquareTransform(),
            Mod7Type.ThreeFive: CubeTransform(),
            Mod7Type.Other: IdentityTransform[int](),
        }
    )

    assert list([] >> powers_mux) == []
    assert list([(Mod7Type.Other, 0)] >> powers_mux) == [0]
    assert list([(Mod7Type.ThreeFive, 3)] >> powers_mux) == [27]
    assert list([(Mod7Type.TwoFour, 4)] >> powers_mux) == [16]
    assert list(
        reversed(list((Mod7Type.from_int(i), i) for i in range(7))) >> powers_mux
    ) == [6, 125, 16, 27, 4, 1, 0]
    assert list(((Mod7Type.from_int(i), i) for i in range(10_000)) >> powers_mux) == [
        i**2 if i % 7 in [2, 4] else (i**3 if i % 7 in [3, 5] else i)
        for i in range(10_000)
    ]


class FizzTransform(IsoTransform[int | str]):
    """A simple transform that squares each integer in the generator."""

    def __call__(self, it: Iterable[int | str]) -> Iterable[int | str]:
        """Yield "fizz" for each item in the input generator."""
        for item in it:
            if isinstance(item, str):
                yield item + "fizz"
            else:
                yield "fizz"


class BuzzTransform(IsoTransform[int | str]):
    """A simple transform that increments each integer in the generator."""

    def __call__(self, it: Iterable[int | str]) -> Iterable[int | str]:
        """Yield "buzz" for each item in the input generator."""
        for item in it:
            if isinstance(item, str):
                yield item + "buzz"
            else:
                yield "buzz"


class FBType(Enum):
    """Enum to represent different integer types for routing."""

    NOOP = 0
    FIZZ = 1
    BUZZ = 2
    FIZZBUZZ = 3

    @staticmethod
    def from_int(value: int) -> "FBType":
        """Convert an integer to the corresponding FBType."""
        if value % 3 == 0 and value % 5 == 0:
            return FBType.FIZZBUZZ
        elif value % 3 == 0:
            return FBType.FIZZ
        elif value % 5 == 0:
            return FBType.BUZZ
        else:
            return FBType.NOOP


def test_mux_fizzbuzz() -> None:
    # Implement FizzBuzz with a Mux.
    fizzbuzz_mux = MuxTransform[FBType, int | str, int | str](
        {
            FBType.NOOP: IdentityTransform[int | str](),
            FBType.FIZZ: FizzTransform(),
            FBType.BUZZ: BuzzTransform(),
            FBType.FIZZBUZZ: FizzTransform() | BuzzTransform(),
        }
    )

    assert list(((FBType.from_int(i), i) for i in range(1, 31)) >> fizzbuzz_mux) == [
        1,
        2,
        "fizz",
        4,
        "buzz",
        "fizz",
        7,
        8,
        "fizz",
        "buzz",
        11,
        "fizz",
        13,
        14,
        "fizzbuzz",
        16,
        17,
        "fizz",
        19,
        "buzz",
        "fizz",
        22,
        23,
        "fizz",
        "buzz",
        26,
        "fizz",
        28,
        29,
        "fizzbuzz",
    ]
