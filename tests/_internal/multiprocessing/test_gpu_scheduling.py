# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import os
import random
import subprocess
import time
from functools import partial
from multiprocessing import Manager
from multiprocessing.managers import ListProxy
from unittest import mock

import pytest

from faith._internal.multiprocessing.gpu_scheduling import (
    GPUJob,
    JobOutcome,
    _AllocatableJob,
    _detect_available_gpus,
    _execute_in_gpu_context,
    _run_in_parallel,
    _set_cuda_visible_devices,
    run_gpu_jobs_in_parallel,
)


@pytest.mark.parametrize(
    "gpu_ids,expected",
    [
        (["0", "1", "2"], "0,1,2"),
        (["3"], "3"),
        ([], ""),
    ],
)
def test_set_cuda_visible_devices(gpu_ids: list[str], expected: str) -> None:
    """Test setting CUDA_VISIBLE_DEVICES with various GPU ID configurations."""
    with mock.patch.dict(os.environ, {}, clear=False):
        _set_cuda_visible_devices(gpu_ids)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == expected


@pytest.mark.parametrize(
    "gpu_ids,return_value",
    [
        (["0", "1"], "test_result"),
        (["2"], 42),
        ([], {"key": "value"}),
    ],
)
def test_execute_in_gpu_context(gpu_ids: list[str], return_value: object) -> None:
    """Test executing a function within a GPU context properly sets environment."""
    mock_fn = mock.Mock(return_value=return_value)

    with mock.patch(
        "faith._internal.multiprocessing.gpu_scheduling._set_cuda_visible_devices"
    ) as mock_set_cuda:
        assert _execute_in_gpu_context(mock_fn, gpu_ids) == return_value

        # Verify that the mock functions were called correctly.
        mock_set_cuda.assert_called_once_with(gpu_ids)
        mock_fn.assert_called_once_with()


@pytest.mark.parametrize(
    "stdout,expected_gpu_ids",
    [
        ("0\n1\n2\n3\n", ["0", "1", "2", "3"]),
        ("0\n", ["0"]),
        ("5\n7\n", ["5", "7"]),
    ],
)
def test_detect_available_gpus_success(
    stdout: str, expected_gpu_ids: list[str]
) -> None:
    """Test successful GPU detection via nvidia-smi."""
    mock_result = mock.Mock(stdout=stdout)

    with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
        assert _detect_available_gpus() == expected_gpu_ids
        mock_run.assert_called_once_with(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )


@pytest.mark.parametrize(
    "exception_type",
    [
        subprocess.CalledProcessError(1, "nvidia-smi"),
        FileNotFoundError("nvidia-smi not found"),
        subprocess.TimeoutExpired("nvidia-smi", 5),
    ],
)
def test_detect_available_gpus_failure(exception_type: Exception) -> None:
    """Test that GPU detection failures raise RuntimeError."""
    with mock.patch("subprocess.run", side_effect=exception_type):
        with pytest.raises(RuntimeError, match="Failed to detect GPUs via nvidia-smi"):
            _detect_available_gpus()


def _resource_counter(resources: list[str]) -> int:
    """A test function that counts the number of allocated resources."""
    return len(resources)


def _failing_job(resources: list[str]) -> int:
    """A test function that raises an exception."""
    raise ValueError("Test error")


def test_run_in_parallel_basic_execution() -> None:
    """Test that jobs execute with correct resource allocation and return results."""
    jobs = [
        _AllocatableJob[int](id="job1", fn=_resource_counter, need=2),
        _AllocatableJob[int](id="failing_job", fn=_failing_job, need=3),
        _AllocatableJob[int](id="job2", fn=_resource_counter, need=1),
        _AllocatableJob[int](id="job3", fn=_resource_counter, need=1),
    ]

    outcomes: list[JobOutcome[int]] = list(_run_in_parallel(jobs, ["0", "1", "2"]))

    assert sorted(outcomes, key=lambda o: o.job_id) == [
        JobOutcome(job_id="failing_job", ok=False, error=mock.ANY, resources=mock.ANY),
        JobOutcome(job_id="job1", ok=True, result=2, resources=mock.ANY),
        JobOutcome(job_id="job2", ok=True, result=1, resources=mock.ANY),
        JobOutcome(job_id="job3", ok=True, result=1, resources=mock.ANY),
    ]


def _test_job_fn() -> str:
    """A test function that returns a simple string result."""
    return "success"


def _test_bad_job_fn() -> str:
    """A test function that raises an exception."""
    raise RuntimeError("Job failed")


def test_run_gpu_jobs_in_parallel_basic_execution() -> None:
    """Test that run_gpu_jobs_in_parallel detects GPUs and executes jobs correctly."""
    jobs = [
        GPUJob[str](id="job1", fn=_test_job_fn, need=2),
        GPUJob[str](id="job2", fn=_test_bad_job_fn, need=1),
        GPUJob[str](id="job3", fn=_test_job_fn, need=4),
        GPUJob[str](id="job4", fn=_test_job_fn, need=1),
    ]

    with mock.patch(
        "faith._internal.multiprocessing.gpu_scheduling._detect_available_gpus",
        return_value=["0", "1", "2", "3"],
    ) as mock_detect:
        outcomes: list[JobOutcome[str]] = list(run_gpu_jobs_in_parallel(jobs))

        mock_detect.assert_called_once_with()
        assert sorted(outcomes, key=lambda o: o.job_id) == [
            JobOutcome(job_id="job1", ok=True, result="success", resources=mock.ANY),
            JobOutcome(job_id="job2", ok=False, error=mock.ANY, resources=mock.ANY),
            JobOutcome(job_id="job3", ok=True, result="success", resources=mock.ANY),
            JobOutcome(job_id="job4", ok=True, result="success", resources=mock.ANY),
        ]


def test_run_gpu_jobs_in_parallel_no_gpus() -> None:
    """Test that run_gpu_jobs_in_parallel raises RuntimeError when no GPUs are detected."""
    with mock.patch(
        "faith._internal.multiprocessing.gpu_scheduling._detect_available_gpus",
        return_value=[],
    ):
        with pytest.raises(
            RuntimeError, match="No GPUs detected for parallel execution"
        ):
            list(
                run_gpu_jobs_in_parallel([GPUJob[str](id="j", fn=_test_job_fn, need=1)])
            )


def _tracking_job(shared_log: ListProxy) -> str:
    """A job that tracks resource usage by reading CUDA_VISIBLE_DEVICES."""
    # Fetch the allocated resources from environment.
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allocated_resources = set(cuda_devices.split(",") if cuda_devices else [])

    # Log the start and end of this job with its allocated resources.
    shared_log.append(("alloc", allocated_resources, time.time()))
    time.sleep(random.uniform(0.01, 0.1))
    try:
        return f"used-{len(allocated_resources)}"
    finally:
        shared_log.append(("free", allocated_resources, time.time()))


def test_run_gpu_jobs_in_parallel_disjoint_resource_usage() -> None:
    """Test that multiple jobs running in parallel use disjoint sets of GPU resources."""
    # Create a shared log to track resource usage across processes.
    manager = Manager()
    shared_log = manager.list()

    # Create jobs that will run concurrently and track their resource usage.
    jobs = [
        GPUJob[str](id="job0", fn=partial(_tracking_job, shared_log), need=1),
        GPUJob[str](id="job1", fn=partial(_tracking_job, shared_log), need=2),
        GPUJob[str](id="job2", fn=partial(_tracking_job, shared_log), need=1),
        GPUJob[str](id="job3", fn=partial(_tracking_job, shared_log), need=3),
        GPUJob[str](id="job4", fn=partial(_tracking_job, shared_log), need=2),
        GPUJob[str](id="job5", fn=partial(_tracking_job, shared_log), need=1),
        GPUJob[str](id="job6", fn=partial(_tracking_job, shared_log), need=1),
        GPUJob[str](id="job7", fn=partial(_tracking_job, shared_log), need=1),
        GPUJob[str](id="job8", fn=partial(_tracking_job, shared_log), need=2),
        GPUJob[str](id="job9", fn=partial(_tracking_job, shared_log), need=1),
    ]

    with mock.patch(
        "faith._internal.multiprocessing.gpu_scheduling._detect_available_gpus",
        return_value=["0", "1", "2"],
    ):
        outcomes: list[JobOutcome[str]] = list(run_gpu_jobs_in_parallel(jobs))
        assert sorted(outcomes, key=lambda o: o.job_id) == [
            JobOutcome(job_id="job0", ok=True, result="used-1", resources=mock.ANY),
            JobOutcome(job_id="job1", ok=True, result="used-2", resources=mock.ANY),
            JobOutcome(job_id="job2", ok=True, result="used-1", resources=mock.ANY),
            JobOutcome(job_id="job3", ok=True, result="used-3", resources=mock.ANY),
            JobOutcome(job_id="job4", ok=True, result="used-2", resources=mock.ANY),
            JobOutcome(job_id="job5", ok=True, result="used-1", resources=mock.ANY),
            JobOutcome(job_id="job6", ok=True, result="used-1", resources=mock.ANY),
            JobOutcome(job_id="job7", ok=True, result="used-1", resources=mock.ANY),
            JobOutcome(job_id="job8", ok=True, result="used-2", resources=mock.ANY),
            JobOutcome(job_id="job9", ok=True, result="used-1", resources=mock.ANY),
        ]

    # Analyze the log to ensure resources were used disjointly.
    # Build a timeline of which resources were in use at each point sorted by timestamp.
    used_resources: set[str] = set()
    for event_type, resources, _ in sorted(shared_log, key=lambda x: x[2]):
        if event_type == "alloc":
            assert not (
                used_resources & resources
            ), f"Resource conflict with {list(used_resources & resources)}"
            used_resources.update(resources)
        elif event_type == "free":
            used_resources.difference_update(resources)
    assert not used_resources, "Some resources were not released after job completion"
