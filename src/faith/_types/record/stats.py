# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from dataclasses_json import DataClassJsonMixin, config

from faith._types.config.patterns import AnswerFormat

# A label or prediction may be any JSON value: a single string answer, a list of
# strings, a numeric target, a structured object. Each score function binds this
# alias to the concrete shape it expects via its ScoreFn type parameter, which
# mypy checks at the call site.
Labeling: TypeAlias = (
    str | int | float | bool | None | Sequence["Labeling"] | Mapping[str, "Labeling"]
)
SingleLabelSeq: TypeAlias = Sequence[str | None]
MultiLabelSeq: TypeAlias = Sequence[Sequence[str] | None]

MetricSummary: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class StatsRecord(DataClassJsonMixin):
    """Statistics computed for a single record by a log grader.

    `error` holds the title of the generation error that produced no answer to
    grade. A failure is not a wrong answer, so an errored record is excluded from
    the accuracy metrics and counted separately.

    `stopped_reason` is agentic-only, and left None by every QA grader.
    """

    label: Labeling | None
    prediction: Labeling | None
    answer_format: AnswerFormat = field(
        metadata=config(decoder=AnswerFormat, encoder=str)
    )
    subject: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    log_probs: dict[str, float] | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    num_output_tokens: int | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    max_token_halt: bool | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    scores: dict[str, Any] | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    error: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    stopped_reason: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
