# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Logger provided as a context manager for collecting records over a benchmark run."""

from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any, Generic, Type, TypeVar, cast

from faith._internal.io.json import write_as_json
from faith._internal.iter.transform import IsoTransform

_LOG_TYPE = TypeVar("_LOG_TYPE")


class LogCollector(Generic[_LOG_TYPE]):
    """A context manager for collecting records over a benchmark run."""

    def __init__(self, log_filename: Path, log_on_exception: bool = False) -> None:
        """Initialize the LogCollector.

        Args:
            log_filename (Path): The path to the log file where logs will be written.
            log_on_exception (bool): If True, logs will be written even if
                an exception occurs; otherwise, logs are only written on a
                successful exit without exceptions.
        """
        self._log_filename = log_filename
        self._log_on_exception = log_on_exception
        self._logs: list[_LOG_TYPE] = []

    def __enter__(self) -> "LogCollector":
        """Enter the runtime context related to this object."""
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context related to this object."""
        if (exc_type is None or self._log_on_exception) and len(self._logs) > 0:
            write_as_json(self._log_filename, self._logs)

    def log(self, entry: _LOG_TYPE) -> None:
        """Log the `entry` to the log file."""
        self._logs.append(entry)


class LoggingTransform(IsoTransform[_LOG_TYPE], Generic[_LOG_TYPE]):
    """A transform that logs items from an iterator to a specified log file."""

    def __init__(self, log_file: Path, **logger_kwargs: Any) -> None:
        """Initialize the LoggingTransform with a log file path."""
        self._log_file = log_file
        self._logger_kwargs = logger_kwargs

    def __call__(self, src: Iterable[_LOG_TYPE]) -> Iterable[_LOG_TYPE]:
        """Log the items from the `src` iterator and yield them."""
        # Construct a one-item cache to delay yielding until the next item arrives.
        # It is initialized to a placeholder value but will be overwritten before use.
        last_item: _LOG_TYPE = cast(_LOG_TYPE, {})  # Placeholder value.
        items_cached = False
        with LogCollector(self._log_file, **self._logger_kwargs) as logger:
            # Iterate through the source but delay yielding until the next item arrives.
            for item in src:
                logger.log(item)
                if items_cached:
                    yield last_item
                last_item = item
                items_cached = True
        # Yield the last item after the loop completes and the logger is closed.
        # This ensures that the logging file is written before yielding the last item,
        # which is critical for ensuring that there are no races if multiple
        # LoggingTransforms are used in sequence.
        # Note: this only works because the LogCollector opens the file and writes
        # the logs only at exit.
        if items_cached:
            yield last_item
