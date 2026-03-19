# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from enum import auto

from faith._internal.iter.join import LeftJoinTransform
from faith._internal.iter.transform import Mapping, Transform
from faith._types.enums import CIStrEnum
from faith._types.record.sample_record import RecordStatus, SampleRecord


class ReplacementStrategy(CIStrEnum):
    """Defines the strategy to use when reconciling matching records."""

    NEVER = auto()
    ALWAYS = auto()
    IF_DATA_HASH_DIFFERS = auto()


class _RecordReconciliation(
    Mapping[tuple[SampleRecord, SampleRecord | None], tuple[RecordStatus, SampleRecord]]
):
    """A transform that reconciles new data with existing data according to a specified strategy."""

    def __init__(self, strategy: ReplacementStrategy):
        self._strategy = strategy

    def _map_fn(
        self, element: tuple[SampleRecord, SampleRecord | None]
    ) -> tuple[RecordStatus, SampleRecord]:
        """Reconcile a pair of (new, existing) records according to the specified strategy."""
        new, existing = element
        if existing is None:
            return (RecordStatus.DIRTY, new)

        if self._strategy == ReplacementStrategy.NEVER:
            return (RecordStatus.CLEAN, existing)
        if self._strategy == ReplacementStrategy.ALWAYS:
            return (RecordStatus.DIRTY, new)
        if self._strategy == ReplacementStrategy.IF_DATA_HASH_DIFFERS:
            assert new.metadata.data_hash is not None
            if (
                existing.metadata.data_hash is None
                or existing.metadata.data_hash != new.metadata.data_hash
            ):
                return (RecordStatus.DIRTY, new)
        return (RecordStatus.CLEAN, existing)


def reconcile_records(
    existing: Iterable[SampleRecord], strategy: ReplacementStrategy
) -> Transform[SampleRecord, tuple[RecordStatus, SampleRecord]]:
    """Creates a transform that reconciles a new record stream with existing records."""
    return LeftJoinTransform(
        existing, on_key=lambda record: record.data.benchmark_sample_hash
    ) | _RecordReconciliation(strategy)
