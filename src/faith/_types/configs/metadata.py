# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark metadata configuration."""

from dataclasses import dataclass, field
from enum import Enum


class BenchmarkState(Enum):
    """Enum for benchmark states."""

    ENABLED = "enabled"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"
    TEST_ONLY = "test_only"

    @staticmethod
    def from_string(s: str) -> "BenchmarkState":
        """Convert a string to a BenchmarkState enum."""
        try:
            return BenchmarkState[s.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown benchmark state: {s}") from e


@dataclass(frozen=True)
class MetadataConfig:
    """Benchmark metadata."""

    name: str | None = None
    description: str | None = None
    license: str | None = None
    urls: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    state: BenchmarkState = BenchmarkState.ENABLED
