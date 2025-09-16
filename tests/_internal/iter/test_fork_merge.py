# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import random
import time
from typing import Iterable

import pytest

from faith._internal.iter.fork_merge import ForkAndMergeTransform


def simple_function(x: int) -> int:
    """A simple function that simulates a computation."""
    return x + 1


def delayed_function(x: int) -> int:
    """A simple function that simulates a slow computation."""
    time.sleep(random.uniform(0, 1 / 1000))  # Simulate variable delay
    return simple_function(x)


@pytest.mark.parametrize(
    "lst, max_workers",
    [
        ([], 5),
        (range(1), 1),
        (range(100), 10),
        (range(1000), 20),
    ],
)
def test_thread_execution_transformer(lst: Iterable[int], max_workers: int) -> None:
    """Test the ForkAndMergeTransform to ensure order is preserved."""
    transformer = ForkAndMergeTransform[int, int](
        transform_fn=delayed_function,
        except_handler=lambda e: -1,
        max_workers=4,
    )
    assert list(lst >> transformer) == list(map(simple_function, lst))


def delay_with_exception(x: int) -> int:
    """A function that raises an exception for a specific input."""
    if x % 7 == 2:
        raise ValueError("Simulated failure for input 5")
    return delayed_function(x)


@pytest.mark.parametrize(
    "lst, max_workers",
    [
        ([], 5),
        (range(1), 1),
        (range(100), 10),
        (range(1000), 20),
    ],
)
def test_thread_execution_transformer_with_exceptions(
    lst: Iterable[int], max_workers: int
) -> None:
    """Test the ForkAndMergeTransform with a simple function."""
    transformer = ForkAndMergeTransform(
        transform_fn=delay_with_exception,
        except_handler=lambda e: "foo",
        max_workers=4,
    )
    assert list(lst >> transformer) == [
        simple_function(x) if x % 7 != 2 else "foo" for x in lst
    ]
