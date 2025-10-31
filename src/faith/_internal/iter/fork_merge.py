# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""A module for executing functions in separate threads."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generic, Iterable, TypeVar

from faith._internal.iter.transform import Transform
from faith._internal.types.collections import SequencedBuffer

_IN = TypeVar("_IN")
_OUT = TypeVar("_OUT")


class ForkAndMergeTransform(Transform[_IN, _OUT], Generic[_IN, _OUT]):
    """A transformer that executes a function in separate threads and merges their results."""

    def __init__(
        self,
        transform_fn: Callable[[_IN], _OUT],
        exception_handler: Callable[[BaseException], _OUT],
        max_workers: int,
    ):
        """Initialize with the underlying transform function and exception handler."""
        self._transform_fn = transform_fn
        self._exception_handler = exception_handler
        self._max_workers = max_workers

    def __call__(self, inputs: Iterable[_IN]) -> Iterable[_OUT]:
        """Execute the function in a separate thread for each request and merge their results."""
        buffer = SequencedBuffer[_OUT]()
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit tasks along with their index.
            futures = {
                executor.submit(self._transform_fn, input_): idx
                for idx, input_ in enumerate(inputs)
            }

            for future in as_completed(futures):
                exception = future.exception()
                if exception:
                    # If an exception occurred, handle it with the provided handler.
                    buffer.add_at(futures[future], self._exception_handler(exception))
                else:
                    # Add the next completed future.
                    buffer.add_at(futures[future], future.result())

                # Yield all available outputs from the buffer.
                while output := buffer.next_in_order():
                    yield output
            assert (
                len(buffer) == 0
            ), "Internal Error: Buffer not fully populated; disparate items remain."
