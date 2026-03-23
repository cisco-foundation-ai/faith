# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark metadata configuration."""

from dataclasses import dataclass, field
from enum import auto

from dataclasses_json import DataClassJsonMixin, config

from faith._types.enums import CIStrEnum


class BenchmarkState(CIStrEnum):
    """Enum for benchmark states."""

    ENABLED = auto()
    EXPERIMENTAL = auto()
    DISABLED = auto()
    TEST_ONLY = auto()


@dataclass(frozen=True)
class MetadataConfig(DataClassJsonMixin):
    """Benchmark metadata."""

    name: str | None = field(default=None, metadata=config(exclude=lambda x: x is None))
    description: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    license: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    urls: list[str] = field(
        default_factory=list, metadata=config(exclude=lambda x: not x)
    )
    categories: list[str] = field(
        default_factory=list, metadata=config(exclude=lambda x: not x)
    )
    state: BenchmarkState = field(
        default=BenchmarkState.ENABLED,
        metadata=config(encoder=str, decoder=BenchmarkState),
    )
