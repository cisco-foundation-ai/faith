# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

from faith._types.records.model_record import ModelRecord
from faith._types.records.prompt_record import PromptRecord
from faith._types.records.sample_record import SampleRecord, _Metadata
from faith._types.records.stats_record import StatsRecord


def make_fake_record(
    *,
    metadata: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    model_data: dict[str, Any] | None = None,
    stats: StatsRecord | None = None,
) -> SampleRecord:
    """Create a fake `SampleRecord` for testing purposes."""
    return SampleRecord(
        metadata=cast(
            _Metadata, {"data_hash": "aaabbf123", "version": "1.0"} | (metadata or {})
        ),
        data=PromptRecord.from_dict(
            {
                "benchmark_sample_index": 0,
                "benchmark_sample_hash": "fffaabb123",
                "subject": None,
                "system_prompt": None,
                "instruction": None,
                "question": "What is 2 + 2?",
                "choices": None,
                "label": None,
                "formatted_question": "What is 2 + 2?",
                "formatted_answer": None,
                "question_prompt": "What is 2 + 2?",
            }
            | (data or {})
        ),
        model_data=ModelRecord.from_dict(
            {
                "prompt": "What is 2 + 2?",
                "answer_symbol_ids": {"A": 0, "B": 1, "C": 2, "D": 3},
            }
            | (model_data or {}),
        ),
        stats=stats,
    )
