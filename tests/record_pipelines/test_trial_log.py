# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Persistence boundaries, log compatibility, and atomic failure behavior."""

import pytest

from faith._internal.algo.hash import dict_sha256
from faith._types.record.agent import StoppedReason
from faith._types.record.model_response import GenerationError
from faith.agentic.core import Task
from faith.record_pipelines.agentic_formatting import TaskFormatter
from faith.record_pipelines.trial_log import (
    checkpoint_records,
    persist_records,
    read_trial_log,
)
from tests._types.record.fake_agent_result_maker import make_fake_agent_result


def _record(index=0, failed=False):
    (record,) = list(
        [(index, Task(id=str(index), prompt="inspect", ground_truth=["a.c"]))]
        >> TaskFormatter("v1", None)
    )
    record.model_data.agent_trajectory = make_fake_agent_result(
        ["a.c"],
        stopped_reason=StoppedReason.ERROR if failed else StoppedReason.SUBMITTED,
        error="failure details" if failed else None,
    )
    if failed:
        record.model_data.error = GenerationError(
            title="Agent Run Error", details="failure details"
        )
    return record


@pytest.mark.parametrize("failed", [False, True])
def test_checkpoint_contains_full_record_before_yield(tmp_path, failed):
    path = tmp_path / "benchmark-log.json"
    record = _record(failed=failed)
    output = iter([record] >> checkpoint_records(path, []))
    assert next(output) is record
    (saved,) = read_trial_log(path)
    assert saved == record  # Messages, tool calls/responses, errors and metadata.
    assert saved.metadata.model_data_hash == dict_sha256(saved.model_data.to_dict())
    assert not list(output)


def test_batch_persistence_is_unchanged_on_interruption(tmp_path):
    path = tmp_path / "benchmark-log.json"
    list([_record(4)] >> persist_records(path))
    previous = path.read_bytes()

    def interrupted():
        yield _record(0)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        list(interrupted() >> persist_records(path))
    assert path.read_bytes() == previous


def test_final_checkpoint_matches_batch_log_format_and_order(tmp_path):
    records = [_record(5), _record(1, failed=True), _record(3)]
    batch, checkpoint = tmp_path / "batch.json", tmp_path / "checkpoint.json"
    list(records >> persist_records(batch))
    list(records >> checkpoint_records(checkpoint, []))
    assert checkpoint.read_bytes() == batch.read_bytes()


def test_failed_final_prune_preserves_valid_checkpoint_and_raises(
    tmp_path, monkeypatch
):
    path = tmp_path / "benchmark-log.json"
    existing = [_record(0), _record(1)]
    list(existing >> persist_records(path))
    selected = _record(1, failed=True)
    output = iter([selected] >> checkpoint_records(path, existing))
    assert next(output) is selected
    previous = path.read_bytes()

    def fail_replace(*_args):
        raise OSError("cannot replace checkpoint")

    monkeypatch.setattr("faith._internal.io.json.os.replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace checkpoint"):
        next(output)
    assert path.read_bytes() == previous
    assert [record.data.benchmark_sample_index for record in read_trial_log(path)] == [
        0,
        1,
    ]
