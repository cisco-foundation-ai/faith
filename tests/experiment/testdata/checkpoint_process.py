# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Disposable subprocess: real trial/persistence pipeline, deterministic fake tasks."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

from faith._internal.io.json import write_as_json
from faith.agentic.core import Task
from tests.experiment.test_trial_runner import _agentic_benchmark, _run_agentic_trial


def main(directory: Path, pause: str) -> None:
    """Pause during task B or its atomic replacement until the parent kills us."""
    log_path = directory / "benchmark-log.json"
    executed: list[str] = []
    real_replace = os.replace

    def block():
        (directory / "ready").write_text(pause, encoding="utf-8")
        # The parent owns the still-open stdin pipe. No exception/finalizer is
        # involved in checkpoint recovery: the process will be forcibly killed.
        sys.stdin.read()

    def before_run(task):
        executed.append(task.id)
        write_as_json(directory / f"started-{pause}.json", executed)
        if task.id == "B" and pause == "task":
            block()

    def replace_checkpoint(source, destination):
        if (
            pause == "write"
            and destination == log_path
            and executed
            and executed[-1] == "B"
        ):
            block()
        real_replace(source, destination)

    benchmark = _agentic_benchmark(
        [Task(id=name, prompt=f"do {name}", ground_truth=[name]) for name in "ABC"]
    )
    with patch("faith._internal.io.json.os.replace", replace_checkpoint):
        _run_agentic_trial(benchmark, log_path, before_run=before_run)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
