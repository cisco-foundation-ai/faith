# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import pytest

from faith._internal.threading.periodic import PeriodicTaskContext


def test_periodic_task_context_verify_calls_task_on_init() -> None:
    """verify=True (default) calls task once during __init__."""
    call_count = 0

    def task() -> None:
        nonlocal call_count
        call_count += 1

    PeriodicTaskContext(task, interval=100, verify=True)
    assert call_count == 1


def test_periodic_task_context_verify_propagates_exception() -> None:
    """verify=True propagates exceptions from the initial task call."""

    def bad_task() -> None:
        raise RuntimeError("init failure")

    with pytest.raises(RuntimeError, match="init failure"):
        PeriodicTaskContext(bad_task, interval=100, verify=True)


def test_periodic_task_context_verify_false_skips_init_call() -> None:
    """verify=False skips the initial task call."""
    call_count = 0

    def task() -> None:
        nonlocal call_count
        call_count += 1

    PeriodicTaskContext(task, interval=100, verify=False)
    assert call_count == 0


def test_periodic_task_context_periodic_task_execution() -> None:
    """Task is called periodically while the context is active."""
    call_count = 0
    event = threading.Event()

    def task() -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            event.set()

    with PeriodicTaskContext(task, interval=0.05, verify=False):
        event.wait(timeout=5)

    # At least 3 periodic calls + 1 final call from __exit__
    assert call_count >= 3


def test_periodic_task_context_final_call_on_exit() -> None:
    """__exit__ performs one final task call."""
    calls: list[str] = []

    def task() -> None:
        calls.append("called")

    with PeriodicTaskContext(task, interval=100, verify=False):
        # No periodic calls should have happened yet (interval is 100s)
        assert len(calls) == 0

    # The final call from __exit__
    assert len(calls) == 1


def test_periodic_task_context_responsive_shutdown() -> None:
    """Context manager exits promptly, not blocked by a long interval."""
    with PeriodicTaskContext(lambda: None, interval=100, verify=False):
        start = time.monotonic()
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"Shutdown took {elapsed:.2f}s, expected < 2s"


def test_periodic_task_context_invalid_interval() -> None:
    """Interval must be positive."""
    with pytest.raises(AssertionError, match="Interval must be positive"):
        PeriodicTaskContext(lambda: None, interval=0)
