# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable

from faith._internal.iter.join import LeftJoinTransform
from faith._internal.iter.transform import Mapping, Transform
from faith._types.model.generation import GenerationMode
from faith._types.record.agent import StoppedReason
from faith._types.record.model import ModelRecord
from faith._types.record.sample import RecordStatus, ReplacementStrategy, SampleRecord


class _RecordReconciliation(
    Mapping[tuple[SampleRecord, SampleRecord | None], SampleRecord]
):
    """A transform that reconciles new data with existing data according to a specified strategy."""

    def __init__(
        self, strategy: ReplacementStrategy, reset: Callable[[ModelRecord], None]
    ):
        self._strategy = strategy
        self._reset = reset

    def _map_fn(
        self, element: tuple[SampleRecord, SampleRecord | None]
    ) -> SampleRecord:
        """Reconcile a pair of (new, existing) records according to the specified strategy."""
        selected_record = self._select_record(*element)
        self._reset(selected_record.model_data)
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
    """A transform that labels each record as FRESH or STALE based on whether it already carries model output."""

    def __init__(self, is_fresh: Callable[[SampleRecord], bool]):
        self._is_fresh = is_fresh

    def _map_fn(self, element: SampleRecord) -> tuple[RecordStatus, SampleRecord]:
        """Label the record as FRESH or STALE based on the presence of model output."""
        if self._is_fresh(element):
            return (RecordStatus.FRESH, element)
        # The model output this record was graded on is about to be replaced, and
        # its stats would otherwise survive: persist_records re-stamps
        # model_data_hash, so the eval-time staleness check sees a match.
        element.stats = None
        return (RecordStatus.STALE, element)


def _mode_is_fresh(mode: GenerationMode) -> Callable[[SampleRecord], bool]:
    """Build a fresh predicate for a QA generation mode.

    A record is fresh once it carries the response field for its mode; otherwise
    it is stale and must be (re)queried.
    """

    def is_fresh(record: SampleRecord) -> bool:
        if mode == GenerationMode.LOGITS:
            return bool(record.model_data.logits)
        if mode == GenerationMode.NEXT_TOKEN:
            return bool(record.model_data.next_token)
        if mode == GenerationMode.CHAT_COMP:
            return bool(record.model_data.chat_comp)
        return False

    return is_fresh


def _agentic_is_fresh(record: SampleRecord) -> bool:
    """An agentic record is fresh once it carries a non-errored trajectory."""
    trajectory = record.model_data.agent_trajectory
    return trajectory is not None and trajectory.stopped_reason is not (
        StoppedReason.ERROR
    )


def _reconciler(
    existing: Iterable[SampleRecord],
    strategy: ReplacementStrategy,
    reset: Callable[[ModelRecord], None],
    is_fresh: Callable[[SampleRecord], bool],
) -> Transform[SampleRecord, tuple[RecordStatus, SampleRecord]]:
    """Left-join the new record stream against existing records, then select and label each pair."""
    return (
        LeftJoinTransform(
            existing,
            on_key=lambda record: (
                record.data.benchmark_sample_index,
                record.data.benchmark_sample_hash,
            ),
        )
        | _RecordReconciliation(strategy, reset)
        | _RecordStatusTransform(is_fresh)
    )


def record_reconciler(
    existing: Iterable[SampleRecord],
    strategy: ReplacementStrategy,
    mode: GenerationMode,
) -> Transform[SampleRecord, tuple[RecordStatus, SampleRecord]]:
    """Creates a transform that reconciles a new QA record stream with existing records."""
    return _reconciler(
        existing,
        strategy,
        lambda model_data: model_data.reset_to_mode(mode),
        _mode_is_fresh(mode),
    )


def agentic_record_reconciler(
    existing: Iterable[SampleRecord],
    strategy: ReplacementStrategy,
) -> Transform[SampleRecord, tuple[RecordStatus, SampleRecord]]:
    """Creates a transform that reconciles a new agentic record stream with existing records.

    Agentic runs store their output in `agent_trajectory` rather than a QA
    generation-mode field, so a record is fresh once it carries a trajectory and
    the selected record is reset to keep only that field. An errored trajectory
    is stale so resume retries it, matching QA's per-mode freshness.
    """
    return _reconciler(
        existing,
        strategy,
        lambda model_data: model_data.reset_to_agentic(),
        _agentic_is_fresh,
    )
