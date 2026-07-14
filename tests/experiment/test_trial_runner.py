# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from faith.benchmark.benchmark import QABenchmark
from faith.experiment.trial_runner import (
    QATrialRunner,
    TrialRunner,
    trial_runner_for,
)

# Note: the per-trial pipeline itself (build_dataset >> format >> query >> log)
# is covered by the individual record_pipelines tests and test_formatting; these
# tests cover the runner dispatch that selects which pipeline to run.


def test_trial_runner_for_returns_qa_runner() -> None:
    """QA benchmarks dispatch to the QA trial runner."""
    runner = trial_runner_for(Mock(spec=QABenchmark))
    assert isinstance(runner, QATrialRunner)
    assert isinstance(runner, TrialRunner)


def test_trial_runner_for_rejects_unknown_benchmark() -> None:
    """A benchmark with no matching runner raises a helpful error."""
    with pytest.raises(ValueError, match="No trial runner available"):
        trial_runner_for(Mock())
