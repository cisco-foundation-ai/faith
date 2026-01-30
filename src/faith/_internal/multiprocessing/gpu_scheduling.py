# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import subprocess
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Generic, Iterable, TypeVar

logger = logging.getLogger(__name__)

_RV = TypeVar("_RV")
Resource = str


@dataclass(frozen=True)
class GPUJob(Generic[_RV]):
    """A job to be executed with the given resource allocation."""

    id: str
    fn: Callable[[], _RV]
    need: int = 1


@dataclass(frozen=True)
class JobOutcome(Generic[_RV]):
    """The outcome from a completed job."""

    job_id: str
    ok: bool
    result: _RV | None = None
    error: BaseException | None = None
    resources: list[Resource] = field(default_factory=list)


@dataclass(frozen=True)
class _AllocatableJob(Generic[_RV]):
    """A job that can be allocated resources for its execution."""

    id: str
    fn: Callable[[list[Resource]], _RV]
    need: int = 1


def run_gpu_jobs_in_parallel(
    jobs: Iterable[GPUJob[_RV]], *, max_workers: int | None = None
) -> Iterable[JobOutcome[_RV]]:
    """Execute jobs in parallel, allocating the available GPUs as job resources.

    There is no guarantee on the order of job completion so outcomes may be yielded
    in any order.

    Args:
        jobs: Iterable of Job instances to execute.
        max_workers: Maximum number of parallel worker processes.

    Yields:
        JobOutcome as each job completes (success or failure).
    """
    # Detect available GPUs
    gpu_ids = _detect_available_gpus()
    if not gpu_ids:
        raise RuntimeError("No GPUs detected for parallel execution")

    # Run jobs with GPU resource allocation
    yield from _run_in_parallel(
        [
            _AllocatableJob[_RV](
                id=job.id,
                fn=partial(_execute_in_gpu_context, job.fn),
                need=job.need,
            )
            for job in jobs
        ],
        resources=gpu_ids,
        max_workers=max_workers,
    )


def _run_in_parallel(
    jobs: Iterable[_AllocatableJob[_RV]],
    resources: Iterable[Resource],
    *,
    max_workers: int | None = None,
) -> Iterable[JobOutcome[_RV]]:
    """Execute jobs in parallel for the given `resources`.

    The jobs are allocated resources as they become available and executed in a
    distributed manner using ProcessPoolExecutor. There is no guarantee on the order
    of job completion so outcomes may be yielded in any order.

    Args:
        jobs: Iterable of _AllocatableJob instances to execute.
        resources: Iterable of available resources for allocation.
        max_workers: Maximum number of parallel worker processes.
    Yields:
        JobOutcome as each job completes.
    """
    pending: list[_AllocatableJob[_RV]] = list(jobs)
    running: dict[Future, tuple[_AllocatableJob[_RV], list[Resource]]] = {}
    available: list[Resource] = list(resources)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        while pending or running:
            # Allocate all possible pending jobs with the available resources.
            still_pending: list[_AllocatableJob] = []
            for job in sorted(pending, key=lambda j: j.need, reverse=True):
                if job.need <= len(available):
                    allocation, available = available[: job.need], available[job.need :]
                    fut = ex.submit(job.fn, allocation)
                    running[fut] = (job, allocation)
                else:
                    still_pending.append(job)
            pending = still_pending

            # Wait for at least one running job to complete, process the results
            # from completed jobs, and release their resources.
            done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                job, allocation = running.pop(fut)
                try:
                    res = fut.result()
                    yield JobOutcome[_RV](
                        job_id=job.id, ok=True, result=res, resources=allocation
                    )
                except BaseException as e:  # pylint: disable=broad-exception-caught
                    yield JobOutcome[_RV](
                        job_id=job.id, ok=False, error=e, resources=allocation
                    )
                finally:
                    available += allocation


def _detect_available_gpus() -> list[Resource]:
    """Query nvidia-smi for available GPU IDs.

    Returns:
        List of GPU device IDs (e.g., ['0', '1', '2', '3']) if GPUs are available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        gpu_ids = [
            name for line in result.stdout.strip().split("\n") if (name := line.strip())
        ]
        logger.info("Detected %d GPUs: %s", len(gpu_ids), str(gpu_ids))
        return gpu_ids
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        raise RuntimeError("Failed to detect GPUs via nvidia-smi") from e


def _execute_in_gpu_context(fn: Callable[[], _RV], gpu_ids: list[Resource]) -> _RV:
    """Execute the function `fn` within a GPU context set.

    This function is executed in a separate process and sets CUDA_VISIBLE_DEVICES
    before calling the underlying function.

    Args:
        fn: The underlying function to execute.
        gpu_ids: A list of GPU device IDs to allocate for this execution.

    Returns:
        The return value from `fn`.
    """
    _set_cuda_visible_devices(gpu_ids)
    return fn()


def _set_cuda_visible_devices(gpu_ids: list[Resource]) -> None:
    """Set CUDA_VISIBLE_DEVICES environment variable for this process.

    Args:
        gpu_ids: List of GPU device IDs to make visible.
    """
    if len(gpu_ids) > 0:
        gpu_str = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        logger.info("Process %d using GPUs: [%s]", os.getpid(), gpu_str)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Process %d running without GPU allocation", os.getpid())
