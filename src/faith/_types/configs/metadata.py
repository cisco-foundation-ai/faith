# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark metadata configuration."""

from dataclasses import dataclass, field
from enum import auto

from faith._types.enums import CIStrEnum


class BenchmarkState(CIStrEnum):
    """Enum for benchmark states."""

    ENABLED = auto()
    EXPERIMENTAL = auto()
    DISABLED = auto()
    TEST_ONLY = auto()


@dataclass(frozen=True)
class MetadataConfig:
    """Benchmark metadata."""

    name: str | None = None
    description: str | None = None
    license: str | None = None
    urls: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    state: BenchmarkState = BenchmarkState.ENABLED
