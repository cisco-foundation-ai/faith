# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for inspecting objects in Python."""
from typing import Any, Sequence


def assert_same_length(
    expected_length: int | None = None, **kwargs: Sequence[Any]
) -> None:
    """Check if all provided sequences have the same length."""
    lengths = {name: len(seq) for name, seq in kwargs.items()}
    if expected_length is not None:
        assert all(
            length == expected_length for length in lengths.values()
        ), f"All sequences must have the expected length {expected_length}. Lengths: {lengths}"
    else:
        assert (
            len(set(lengths.values())) <= 1
        ), f"All sequences must have the same length. Lengths: {lengths}"
