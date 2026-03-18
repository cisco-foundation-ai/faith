# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration types for output processing and scoring functions."""

from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import DataClassJsonMixin, config

from faith._types.configs.patterns import PatternDef


@dataclass(frozen=True)
class ScoreFnConfig(DataClassJsonMixin):
    """Configuration for a domain-specific scoring function.

    The ``type`` field identifies which scoring function to use. All remaining
    fields are type-specific kwargs stored in ``kwargs``.
    """

    type: str
    kwargs: dict[str, Any] = field(default_factory=dict)


def _encode_score_fns(val: dict[str, ScoreFnConfig]) -> dict[str, dict[str, Any]]:
    return {k: {"type": v.type, **v.kwargs} for k, v in val.items()}


def _decode_score_fn(val: dict[str, Any] | ScoreFnConfig) -> ScoreFnConfig:
    if isinstance(val, ScoreFnConfig):
        return val
    return ScoreFnConfig(
        type=val["type"],
        kwargs={k: v for k, v in val.items() if k != "type"},
    )


def _decode_score_fns(val: dict[str, Any]) -> dict[str, ScoreFnConfig]:
    return {k: _decode_score_fn(v) for k, v in val.items()}


@dataclass(frozen=True)
class OutputProcessingConfig(DataClassJsonMixin):
    """Configuration for output processing and answer extraction."""

    primary_metric: str | None = None
    answer_formats: list[PatternDef] = field(default_factory=list)
    score_fns: dict[str, ScoreFnConfig] = field(
        default_factory=dict,
        metadata=config(encoder=_encode_score_fns, decoder=_decode_score_fns),
    )
