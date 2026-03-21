# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._internal.algo.hash import dict_sha256
from faith._internal.iter.transform import Mapping, Transform
from faith._types.record.sample import RecordStatus, ReplacementStrategy, SampleRecord


class _CullStaleStats(Mapping[SampleRecord, SampleRecord]):
    """A transform that clears stats based on the replacement strategy."""

    def __init__(self, strategy: ReplacementStrategy):
        self._strategy = strategy

    def _map_fn(self, element: SampleRecord) -> SampleRecord:
        """Clear stats if the strategy requires recomputation."""
        if self._strategy == ReplacementStrategy.ALWAYS:
            element.stats = None
        elif self._strategy == ReplacementStrategy.IF_HASH_DIFFERS:
            if self._hash_differs(element):
                element.stats = None
        return element

    @staticmethod
    def _hash_differs(record: SampleRecord) -> bool:
        """Check if either the data_hash or model_data_hash differ from the current data.

        Records missing a stored hash are treated as stale (hash differs).
        """
        if record.metadata.data_hash is None or record.metadata.model_data_hash is None:
            return True
        return (record.metadata.data_hash != dict_sha256(record.data.to_dict())) or (
            record.metadata.model_data_hash != dict_sha256(record.model_data.to_dict())
        )


class _TagForGrading(Mapping[SampleRecord, tuple[RecordStatus, SampleRecord]]):
    """A transform that labels each record as FRESH or STALE based on whether it has stats."""

    def _map_fn(self, element: SampleRecord) -> tuple[RecordStatus, SampleRecord]:
        status = RecordStatus.FRESH if element.stats else RecordStatus.STALE
        return (status, element)


def stats_screener(
    strategy: ReplacementStrategy,
) -> Transform[SampleRecord, tuple[RecordStatus, SampleRecord]]:
    """Creates a transform that screens records for stale or missing stats."""
    return _CullStaleStats(strategy) | _TagForGrading()
