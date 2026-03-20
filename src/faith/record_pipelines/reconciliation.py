# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from faith._internal.iter.join import LeftJoinTransform
from faith._internal.iter.transform import Mapping, Transform
from faith._types.model.generation import GenerationMode
from faith._types.record.sample_record import (
    RecordStatus,
    ReplacementStrategy,
    SampleRecord,
)


class _RecordReconciliation(
    Mapping[tuple[SampleRecord, SampleRecord | None], SampleRecord]
):
    """A transform that reconciles new data with existing data according to a specified strategy."""

    def __init__(self, strategy: ReplacementStrategy, mode: GenerationMode):
        self._strategy = strategy
        self._mode = mode

    def _map_fn(
        self, element: tuple[SampleRecord, SampleRecord | None]
    ) -> SampleRecord:
        """Reconcile a pair of (new, existing) records according to the specified strategy."""
        selected_record = self._select_record(*element)
        selected_record.model_data.reset_to_mode(self._mode)
        return selected_record

    def _select_record(
        self, new: SampleRecord, existing: SampleRecord | None
    ) -> SampleRecord:
        """Select the appropriate record based on the reconciliation strategy."""
        if existing is None:
            return new

        if self._strategy == ReplacementStrategy.ALWAYS:
            return new
        if self._strategy == ReplacementStrategy.IF_HASH_DIFFERS:
            assert new.metadata.data_hash is not None
            if (
                existing.metadata.data_hash is None
                or existing.metadata.data_hash != new.metadata.data_hash
            ):
                return new

        return existing


class _RecordStatusTransform(Mapping[SampleRecord, tuple[RecordStatus, SampleRecord]]):
    """A transform that labels each record as FRESH or STALE based on whether it has the required model response field."""

    def __init__(self, mode: GenerationMode):
        self._mode = mode

    def _map_fn(self, element: SampleRecord) -> tuple[RecordStatus, SampleRecord]:
        """Label the record as FRESH or STALE based on the presence of the required model response field."""
        return (self._record_status(element), element)

    def _record_status(self, record: SampleRecord) -> RecordStatus:
        """Determine the status of a record based on the presence of the required model response field."""
        if self._mode == GenerationMode.LOGITS:
            return (
                RecordStatus.FRESH if record.model_data.logits else RecordStatus.STALE
            )
        if self._mode == GenerationMode.NEXT_TOKEN:
            return (
                RecordStatus.FRESH
                if record.model_data.next_token
                else RecordStatus.STALE
            )
        if self._mode == GenerationMode.CHAT_COMP:
            return (
                RecordStatus.FRESH
                if record.model_data.chat_comp
                else RecordStatus.STALE
            )

        return RecordStatus.STALE


def record_reconciler(
    existing: Iterable[SampleRecord],
    strategy: ReplacementStrategy,
    mode: GenerationMode,
) -> Transform[SampleRecord, tuple[RecordStatus, SampleRecord]]:
    """Creates a transform that reconciles a new record stream with existing records."""
    return (
        LeftJoinTransform(
            existing, on_key=lambda record: record.data.benchmark_sample_hash
        )
        | _RecordReconciliation(strategy, mode)
        | _RecordStatusTransform(mode)
    )
