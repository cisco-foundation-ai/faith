# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration types for output processing and scoring functions."""

from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import DataClassJsonMixin, config

from faith._types.config.patterns import PatternDef


@dataclass(frozen=True)
class ScoreFnConfig(DataClassJsonMixin):
    """Configuration for a domain-specific scoring function.

    Serialized as a flat dict: ``{"type": "<name>", ...kwargs}``.
    The ``type`` field identifies which scoring function to use. All remaining
    fields are type-specific kwargs stored in ``kwargs``.
    """

    type: str
    kwargs: dict[str, Any] = field(
        default_factory=dict, metadata=config(exclude=lambda x: not x)
    )

    @classmethod
    # pylint: disable=unused-argument
    def from_dict(
        cls, kvs: dict[str, Any], infer_missing: bool = False
    ) -> "ScoreFnConfig":
        """Decode from the flat format: ``{"type": "<name>", ...kwargs}``."""
        return cls(
            type=kvs["type"],
            kwargs={k: v for k, v in kvs.items() if k != "type"},
        )

    # pylint: disable-next=unused-argument
    def to_dict(self, encode_json: bool = False) -> dict[str, Any]:
        """Encode to the flat format: ``{"type": "<name>", ...kwargs}``."""
        return {"type": self.type, **self.kwargs}


@dataclass(frozen=True)
class OutputProcessingConfig(DataClassJsonMixin):
    """Configuration for output processing and answer extraction."""

    primary_metric: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    answer_formats: list[PatternDef] = field(
        default_factory=list, metadata=config(exclude=lambda x: not x)
    )
    score_fns: dict[str, ScoreFnConfig] = field(
        default_factory=dict,
        metadata=config(
            encoder=lambda val: {k: v.to_dict() for k, v in val.items()},
            decoder=lambda val: {k: ScoreFnConfig.from_dict(v) for k, v in val.items()},
            exclude=lambda x: not x,
        ),
    )
