# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, StrEnum, auto
from typing import Any, Iterable

from faith._internal.iter.join import LeftJoinTransform
from faith._internal.iter.transform import Transform


class ReplacementStrategy(StrEnum):
    """Defines the strategy to use when reconciling matching records."""

    NEVER = auto()
    ALWAYS = auto()
    IF_DATA_HASH_DIFFERS = auto()


class RecordStatus(Enum):
    """Indicates whether a record is clean (unchanged) or dirty (new or updated)."""

    CLEAN = auto()
    DIRTY = auto()


class _RecordReconciliation(
    Transform[
        tuple[dict[str, Any], dict[str, Any] | None],
        tuple[RecordStatus, dict[str, Any]],
    ]
):
    """A transform that reconciles new data with existing data according to a specified strategy."""

    def __init__(self, strategy: ReplacementStrategy):
        self._strategy = strategy

    def __call__(
        self,
        src: Iterable[tuple[dict[str, Any], dict[str, Any] | None]],
    ) -> Iterable[tuple[RecordStatus, dict[str, Any]]]:
        """Reconcile each pair of (new, existing) records according to the specified strategy."""
        for new, existing in src:
            yield self._reconcile_matching_records(new, existing)

    def _reconcile_matching_records(
        self,
        new: dict[str, Any],
        existing: dict[str, Any] | None,
    ) -> tuple[RecordStatus, dict[str, Any]]:
        """Reconcile two records according to the specified strategy."""
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
    existing: Iterable[dict[str, Any]],
    strategy: ReplacementStrategy,
) -> Transform[dict[str, Any], tuple[RecordStatus, dict[str, Any]]]:
    """Creates a transform that reconciles a new record stream with existing records."""
    return LeftJoinTransform(
        existing, on_key=lambda record: record["data"]["benchmark_sample_hash"]
    ) | _RecordReconciliation(strategy)
