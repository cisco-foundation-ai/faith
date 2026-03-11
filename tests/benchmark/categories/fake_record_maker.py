# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

from faith._internal.records.types import Record, _Metadata, _ModelData
from faith._types.records.prompt_record import PromptRecord


def make_fake_record(
    *,
    metadata: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    model_data: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> Record:
    """Create a fake `Record` for testing purposes."""
    return Record(
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
        model_data=cast(
            _ModelData,
            {
                "prompt": "What is 2 + 2?",
                "answer_symbol_ids": {"A": 0, "B": 1, "C": 2, "D": 3},
            }
            | (model_data or {}),
        ),
        stats=stats,
    )
