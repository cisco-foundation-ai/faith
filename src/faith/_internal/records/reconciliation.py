# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from enum import StrEnum, auto

from faith._internal.iter.join import LeftJoinTransform
from faith._internal.iter.transform import Mapping, Transform
from faith._internal.records.types import Record, RecordStatus


class ReplacementStrategy(StrEnum):
    """Defines the strategy to use when reconciling matching records."""

    NEVER = auto()
    ALWAYS = auto()
    IF_DATA_HASH_DIFFERS = auto()


class _RecordReconciliation(
    Mapping[tuple[Record, Record | None], tuple[RecordStatus, Record]]
):
    """A transform that reconciles new data with existing data according to a specified strategy."""

    def __init__(self, strategy: ReplacementStrategy):
        self._strategy = strategy

    def _map_fn(
        self, element: tuple[Record, Record | None]
    ) -> tuple[RecordStatus, Record]:
        """Reconcile a pair of (new, existing) records according to the specified strategy."""
        new, existing = element
        if existing is None:
            return (RecordStatus.DIRTY, new)

        if self._strategy == ReplacementStrategy.NEVER:
            return (RecordStatus.CLEAN, existing)
        if self._strategy == ReplacementStrategy.ALWAYS:
            return (RecordStatus.DIRTY, new)
        if (
            self._strategy == ReplacementStrategy.IF_DATA_HASH_DIFFERS
            and existing["metadata"]["data_hash"] != new["metadata"]["data_hash"]
        ):
            return (RecordStatus.DIRTY, new)
        return (RecordStatus.CLEAN, existing)


def reconcile_records(
    existing: Iterable[Record], strategy: ReplacementStrategy
) -> Transform[Record, tuple[RecordStatus, Record]]:
    """Creates a transform that reconciles a new record stream with existing records."""
    return LeftJoinTransform(
        existing, on_key=lambda record: record["data"]["benchmark_sample_hash"]
    ) | _RecordReconciliation(strategy)
