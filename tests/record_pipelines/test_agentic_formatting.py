# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the task to log-record mapping."""

from pathlib import Path
from typing import Any

from faith._internal.algo.hash import dict_sha256
from faith._types.json_types import JsonValue
from faith._types.record.sample import SampleRecord
from faith.agentic.core.task import Task
from faith.record_pipelines.agentic_formatting import TaskFormatter

_PROMPT = "Localize the vulnerability."
_SYSTEM_PROMPT = "You are a security analyst."


def _task(
    ground_truth: JsonValue,
    metadata: dict[str, JsonValue] | None = None,
    data_path: Path | None = None,
    environment_config: dict[str, JsonValue] | None = None,
    data_identity: JsonValue | None = None,
) -> Task:
    return Task(
        id="task-1",
        prompt=_PROMPT,
        ground_truth=ground_truth,
        metadata=metadata or {},
        data_path=data_path,
        environment_config=environment_config or {},
        data_identity=data_identity,
    )


def _format(
    task: Task, *, index: int = 0, system_prompt: str | None = _SYSTEM_PROMPT
) -> SampleRecord:
    (record,) = list(iter({index: task}.items()) >> TaskFormatter("v1", system_prompt))
    return record


def test_maps_task_into_bare_record() -> None:
    """Test the mapping sets data/metadata and the label, leaves the trajectory and stats None."""
    record = _format(_task(["a.c", "b.c"], metadata={"difficulty": "easy"}), index=3)

    expected_data: dict[str, Any] = {
        "benchmark_sample_index": 3,
        "benchmark_sample_hash": dict_sha256(
            {
                "id": "task-1",
                "prompt": _PROMPT,
                "ground_truth": ["a.c", "b.c"],
                "data_identity": None,
                "environment_config": {},
            }
        ),
        "subject": None,
        "system_prompt": _SYSTEM_PROMPT,
        "instruction": None,
        "question": _PROMPT,
        "choices": None,
        "label": ["a.c", "b.c"],
        "formatted_question": _PROMPT,
        "formatted_answer": None,
        "question_prompt": _PROMPT,
        "ancillary_data": {"difficulty": "easy"},
    }
    # No "agent_trajectory" key: the record is bare until AgentLoopRunner fills it.
    assert record.to_dict() == {
        "metadata": {
            "version": "v1",
            "data_hash": dict_sha256(expected_data),
        },
        "data": expected_data,
        "model_data": {
            "prompt": _PROMPT,
            "answer_symbol_ids": {},
        },
    }
    assert record.stats is None


def test_ground_truth_becomes_label_across_json_shapes() -> None:
    """Test a non-string ground truth (now a valid Labeling) flows into the label."""
    record = _format(_task({"file": "a.c", "line": 42}))
    assert record.data.label == {"file": "a.c", "line": 42}


def test_empty_metadata_yields_no_ancillary_data() -> None:
    """Test empty task metadata is dropped rather than stored as an empty dict."""
    record = _format(_task("flag{x}"))
    assert record.data.ancillary_data is None
    assert "ancillary_data" not in record.data.to_dict()


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

    hashes = {_format(task).data.benchmark_sample_hash for task in tasks}
    assert len(hashes) == len(tasks)


def test_sample_hash_ignores_the_runtime_cache_root() -> None:
    """The same payload reconciles even when fetched into different cache roots."""
    identity: JsonValue = {"repo": "acme/widget", "sha": "a" * 40, "md5": "0" * 32}
    tasks = [
        _task("flag{x}", data_path=Path("/cache-a/a.zip"), data_identity=identity),
        _task("flag{x}", data_path=Path("/cache-b/a.zip"), data_identity=identity),
    ]

    hashes = {_format(task).data.benchmark_sample_hash for task in tasks}
    assert len(hashes) == 1


def test_data_hash_covers_the_system_prompt() -> None:
    """A changed system prompt is a changed record, so IF_HASH_DIFFERS re-runs the task.

    The system prompt is not part of benchmark_sample_hash, so data_hash is the
    only thing that distinguishes two runs configured with different standing
    instructions.
    """
    prompts = [None, _SYSTEM_PROMPT, "You are a helpful assistant."]

    hashes = {
        _format(_task("flag{x}"), system_prompt=prompt).metadata.data_hash
        for prompt in prompts
    }
    assert len(hashes) == len(prompts)
