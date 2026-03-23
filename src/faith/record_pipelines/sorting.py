# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from functools import reduce
from typing import Any, Generic, Protocol, TypeVar

from faith._internal.iter.transform import IsoTransform
from faith._types.record.sample import SampleRecord


class _Comparable(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


_T = TypeVar("_T", bound=_Comparable)


class SortByTransform(IsoTransform[SampleRecord], Generic[_T]):
    """A transform that sorts an iterable of SampleRecords by a field path."""

    def __init__(self, *fields: str) -> None:
        """Initialize with a sequence of field names (e.g. 'data', 'prompt')."""
        self._attrs = fields

    def _resolve(self, record: SampleRecord) -> _T:
        """Resolve the field path to a value on the record."""
        return reduce(getattr, self._attrs, record)

    def __call__(self, src: Iterable[SampleRecord]) -> Iterable[SampleRecord]:
        """Sort the source iterable by the configured field path."""
        return sorted(src, key=self._resolve)
