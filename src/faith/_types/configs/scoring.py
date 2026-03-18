# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration types for output processing and scoring functions."""

from dataclasses import dataclass, field
from typing import Any

from faith._types.configs.patterns import PatternDef


@dataclass(frozen=True)
class ScoreFnConfig:
    """Configuration for a domain-specific scoring function.

    The ``type`` field identifies which scoring function to use. All remaining
    fields are type-specific kwargs stored in ``kwargs``.
    """

    type: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputProcessingConfig:
    """Configuration for output processing and answer extraction."""

    primary_metric: str | None = None
    answer_formats: list[PatternDef] = field(default_factory=list)
    score_fns: dict[str, ScoreFnConfig] = field(default_factory=dict)
