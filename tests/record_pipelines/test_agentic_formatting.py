# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agent-result to log-record mapping."""

from pathlib import Path
from typing import Any

import pytest

from faith._internal.algo.hash import dict_sha256
from faith._types.json_types import JsonValue
from faith._types.record.agent import StoppedReason
from faith._types.record.model_response import GenerationError
from faith.agentic.core.task import Task
from faith.record_pipelines.agentic_formatting import (
    agent_result_to_sample_record,
)
from tests._types.record.fake_agent_result_maker import (
    make_fake_agent_result as _result,
)

_PROMPT = "Localize the vulnerability."


def _task(
    ground_truth: JsonValue,
    metadata: dict[str, JsonValue] | None = None,
    data_path: Path | None = None,
    environment_config: dict[str, JsonValue] | None = None,
) -> Task:
    return Task(
        id="task-1",
        prompt=_PROMPT,
        ground_truth=ground_truth,
        metadata=metadata or {},
        data_path=data_path,
        environment_config=environment_config or {},
    )


def test_maps_result_into_stale_record() -> None:
    """Test the mapping embeds the trajectory, sets the label, and leaves stats None."""
    task = _task(["a.c", "b.c"], metadata={"difficulty": "easy"})
    result = _result(["a.c", "b.c"])

    record = agent_result_to_sample_record(task, result, index=3, version="v1")

    expected_data: dict[str, Any] = {
        "benchmark_sample_index": 3,
        "benchmark_sample_hash": dict_sha256(
            {
                "id": "task-1",
                "prompt": _PROMPT,
                "ground_truth": ["a.c", "b.c"],
                "data_path": None,
                "environment_config": {},
            }
        ),
        "subject": None,
        "system_prompt": None,
        "instruction": None,
        "question": _PROMPT,
        "choices": None,
        "label": ["a.c", "b.c"],
        "formatted_question": _PROMPT,
        "formatted_answer": None,
        "question_prompt": _PROMPT,
        "ancillary_data": {"difficulty": "easy"},
    }
    assert record.to_dict() == {
        "metadata": {
            "version": "v1",
            "data_hash": dict_sha256(expected_data),
        },
        "data": expected_data,
        "model_data": {
            "prompt": _PROMPT,
            "answer_symbol_ids": {},
            "agent_trajectory": result.to_dict(),
        },
    }
    assert record.stats is None


def test_ground_truth_becomes_label_across_json_shapes() -> None:
    """Test a non-string ground truth (now a valid Labeling) flows into the label."""
    record = agent_result_to_sample_record(
        _task({"file": "a.c", "line": 42}),
        _result(None),
        index=0,
        version="v1",
    )
    assert record.data.label == {"file": "a.c", "line": 42}


def test_empty_metadata_yields_no_ancillary_data() -> None:
    """Test empty task metadata is dropped rather than stored as an empty dict."""
    record = agent_result_to_sample_record(
        _task("flag{x}"), _result("flag{x}"), index=0, version="v1"
    )
    assert record.data.ancillary_data is None
    assert "ancillary_data" not in record.data.to_dict()


def test_an_errored_run_records_a_generation_error() -> None:
    """Test the harness failure reaches the grader as QA's generation error."""
    record = agent_result_to_sample_record(
        _task("flag{x}"),
        _result(None, StoppedReason.ERROR, error="container died"),
        index=0,
        version="v1",
    )
    assert record.model_data.error == GenerationError(
        title="Agent Run Error", details="container died"
    )


@pytest.mark.parametrize(
    "stopped_reason",
    [
        StoppedReason.SUBMITTED,
        StoppedReason.NO_TOOL_CALLS,
        StoppedReason.MAX_CALLS,
        StoppedReason.MAX_TURNS,
        StoppedReason.TIMEOUT,
    ],
)
def test_a_normal_termination_records_no_error(stopped_reason: StoppedReason) -> None:
    # Hitting a limit is an incomplete answer, not a failure: marking it errored
    # would withhold it from the accuracy metrics it belongs in.
    record = agent_result_to_sample_record(
        _task("flag{x}"), _result(None, stopped_reason), index=0, version="v1"
    )
    assert record.model_data.error is None


def test_sample_hash_covers_the_data_payload_and_environment() -> None:
    """A task whose payload or environment changed is a different sample.

    The hash is the reconciliation key: if it ignored these, a re-run against a
    new image or a new payload would reuse the cached trajectory.
    """
    tasks = [
        _task("flag{x}"),
        _task("flag{x}", data_path=Path("/data/cve-2021-1.tar")),
        _task("flag{x}", environment_config={"image": "ubuntu:22.04"}),
    ]

    hashes = {
        agent_result_to_sample_record(
            task, _result("flag{x}"), index=0, version="v1"
        ).data.benchmark_sample_hash
        for task in tasks
    }
    assert len(hashes) == len(tasks)
