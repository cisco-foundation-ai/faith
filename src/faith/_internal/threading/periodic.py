# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""A generic periodic task runner as a context manager."""

import logging
import threading
import time
from collections.abc import Callable
from types import TracebackType

logger = logging.getLogger(__name__)


class PeriodicTaskContext:
    """A context manager that runs a callable at a fixed interval in a background thread."""

    def __init__(
        self, task: Callable[[], None], interval: float, *, verify: bool = True
    ):
        """Initialize the periodic task runner.

        Args:
            task: Zero-arg callable to run periodically.
            interval: Seconds between calls.
            verify: If True, call task() once in __init__ to validate;
                exception propagates on failure.
        """
        assert interval > 0, "Interval must be positive."
        self._task = task
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        if verify:
            self._task()

    def _run(self) -> None:
        """Background loop: wait for interval (minus task duration), then call task."""
        elapsed = 0.0
        while not self._stop_event.wait(timeout=max(0, self._interval - elapsed)):
            start = time.monotonic()
            self._task()
            elapsed = time.monotonic() - start

    def __enter__(self) -> None:
        """Start the background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the background thread and run the task one final time."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._task()
