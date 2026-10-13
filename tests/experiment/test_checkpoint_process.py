# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Recover from process death using a local log shared by separate processes."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from faith._internal.algo.hash import dict_sha256
from faith._internal.io.json import read_json_file
from faith.record_pipelines.trial_log import read_trial_log


@pytest.mark.parametrize("pause", ["task", "write"])
def test_killed_process_resumes_only_uncheckpointed_tasks(tmp_path, pause):
    """Kill during B, or just before its checkpoint rename; A must survive both."""
    root = Path(__file__).parents[2]
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])
    }
    command = [
        sys.executable,
        str(Path(__file__).parent / "testdata/checkpoint_process.py"),
        str(tmp_path),
    ]
    path = tmp_path / "benchmark-log.json"
    with subprocess.Popen(
        command + [pause],
        cwd=root,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            deadline = time.monotonic() + 30
            while not (tmp_path / "ready").exists():
                if process.poll() is not None:
                    pytest.fail(
                        f"Worker exited before pausing: {process.communicate()}"
                    )
                if time.monotonic() >= deadline:
                    pytest.fail("Worker did not reach the interruption boundary")
                time.sleep(0.02)
            (saved,) = read_trial_log(path)
            assert saved.model_data.agent_trajectory is not None
            assert saved.model_data.agent_trajectory.task_id == "A"
            assert saved.metadata.model_data_hash == dict_sha256(
                saved.model_data.to_dict()
            )
            before_kill = path.read_bytes()
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
        assert process.returncode != 0
    assert path.read_bytes() == before_kill
    assert read_json_file(tmp_path / f"started-{pause}.json") == ["A", "B"]

    # A completely new interpreter shares only the same persistent directory.
    resumed = subprocess.run(
        command + ["resume"],
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert read_json_file(tmp_path / "started-resume.json") == ["B", "C"]
    complete = list(read_trial_log(path))
    assert [record.data.benchmark_sample_index for record in complete] == [0, 1, 2]
    assert [
        record.model_data.agent_trajectory.task_id
        for record in complete
        if record.model_data.agent_trajectory is not None
    ] == [
        "A",
        "B",
        "C",
    ]
