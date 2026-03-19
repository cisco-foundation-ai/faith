# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import DataClassJsonMixin, config


@dataclass(frozen=True)
class PromptRecord(DataClassJsonMixin):
    """Base class for benchmark examples."""

    # Metadata about the benchmark sample.
    benchmark_sample_index: int
    benchmark_sample_hash: str
    subject: str | None

    # Components that make up the question.
    system_prompt: str | None
    instruction: str | None
    question: str
    choices: dict[str, str] | None  # Maps symbols (e.g., 'A', 'B') to their choice.
    label: str | None  # aka the "answer" or "ground truth".

    # Formatted question and answer.
    formatted_question: str
    formatted_answer: str | None

    # The full question that is passed to the model.
    question_prompt: str

    # Any additional data associated with this example that is stored alongside it
    # for context or as part of subsequent metric computations.
    ancillary_data: dict[str, Any] | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
