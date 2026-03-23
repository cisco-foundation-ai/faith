# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from enum import Enum
from threading import Thread
from typing import Generic, TypeVar

from faith._internal.collections.queue_iterable import QueueIterable
from faith._internal.iter.transform import Transform

# Generic I/O TypeVars for the MuxTransform.
_KIND = TypeVar("_KIND", bound=Enum)
_IN = TypeVar("_IN")
_OUT = TypeVar("_OUT")


def _run_worker(
    transform: Transform[_IN, _OUT],
    input_qi: QueueIterable[_IN],
    output_qi: QueueIterable[_OUT],
) -> None:
    """Run a transform in a worker thread, pushing results to the output queue."""
    with output_qi:
        for item in input_qi >> transform:
            output_qi.put(item)


class MuxTransform(Transform[tuple[_KIND, _IN], _OUT], Generic[_KIND, _IN, _OUT]):
    """A transform that routes input items to different transforms based on their kind."""

    def __init__(self, transform_map: dict[_KIND, Transform[_IN, _OUT]]):
        """Initialize with a mapping of kinds to transforms."""
        self._transform_map = transform_map

    def __call__(self, src: Iterable[tuple[_KIND, _IN]]) -> Iterable[_OUT]:
        output_qi: QueueIterable[_OUT] = QueueIterable(
            num_producers=len(self._transform_map),
        )
        threads: list[Thread] = []
        input_queues: dict[_KIND, QueueIterable[_IN]] = {}

        with ExitStack() as stack:
            # Create a worker thread for each kind in the transform map.
            for kind, transform in self._transform_map.items():
                qi: QueueIterable[_IN] = QueueIterable()
                stack.enter_context(qi)
                input_queues[kind] = qi
                t = Thread(target=_run_worker, args=(transform, qi, output_qi))
                t.start()
                threads.append(t)

            # Dispatch each item to the appropriate kind's input queue.
            for kind, item in src:
                assert kind in input_queues, f"No transform for kind: {kind}"
                input_queues[kind].put(item)
            # ExitStack closes all input QueueIterables, signalling workers to finish.

        # Yield results as they arrive until all workers have closed the output.
        yield from output_qi

        for t in threads:
            t.join()
