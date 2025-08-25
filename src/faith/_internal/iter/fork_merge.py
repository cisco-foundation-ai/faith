# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""A module for executing functions in separate threads."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generic, Iterable, TypeVar

from faith._internal.iter.transform import Transform
from faith._internal.types.collections import SequencedBuffer

RQST = TypeVar("RQST")
RESP = TypeVar("RESP")


class ForkAndMergeTransform(Transform[RQST, RESP], Generic[RQST, RESP]):
    """A transformer that executes a function in separate threads and merges their results."""

    def __init__(
        self,
        func: Callable[[RQST], RESP],
        except_handler: Callable[[BaseException], RESP],
        max_workers: int,
    ):
        """Initialize with a function to apply, exception handler, and execution parameters."""
        self._func = func
        self._except_handler = except_handler
        self._max_workers = max_workers

    def __call__(self, requests: Iterable[RQST]) -> Iterable[RESP]:
        """Execute the function in a separate thread for each request and merge their results."""
        buffer = SequencedBuffer[RESP]()
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit tasks along with their index
            futures = {
                executor.submit(self._func, req): idx
                for idx, req in enumerate(requests)
            }

            for future in as_completed(futures):
                exception = future.exception()
                if exception:
                    # If an exception occurred, handle it with the provided handler.
                    buffer.add_at(futures[future], self._except_handler(exception))
                else:
                    # Add the next completed future.
                    buffer.add_at(futures[future], future.result())

                # Yield all available outputs from the buffer.
                while output := buffer.next_in_order():
                    yield output
            assert (
                len(buffer) == 0
            ), "Internal Error: Buffer not fully populated; disparate items remain."
