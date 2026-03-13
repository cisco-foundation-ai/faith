# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from dataclasses_json import DataClassJsonMixin, config

from faith._internal.algo.matching import AnswerFormat

Labeling: TypeAlias = str | Sequence[str]
SingleLabelSeq: TypeAlias = Sequence[str | None]
MultiLabelSeq: TypeAlias = Sequence[Sequence[str] | None]


@dataclass(frozen=True)
class StatsRecord(DataClassJsonMixin):
    """Statistics computed for a single record by a log grader."""

    label: Labeling | None
    prediction: Labeling | None
    answer_format: AnswerFormat = field(
        metadata=config(decoder=AnswerFormat.from_string, encoder=str)
    )
    subject: str | None = None
    log_probs: dict[str, float] | None = None
    num_output_tokens: int | None = None
    max_token_halt: bool | None = None
    scores: dict[str, Any] | None = None
