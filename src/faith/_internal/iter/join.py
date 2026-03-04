# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides a join transform for combining two iterables by key."""

from collections import defaultdict
from collections.abc import Hashable, Iterable
from typing import Callable, Generic, TypeVar

from faith._internal.iter.transform import Transform

_IN = TypeVar("_IN")
_OUT = TypeVar("_OUT")
_KEY = TypeVar("_KEY", bound=Hashable)


def _add_key(s: set[_KEY], k: _KEY) -> _KEY:
    """Add a key to a set as a side effect and return the key."""
    s.add(k)
    return k


class JoinTransform(Transform[_IN, _OUT], Generic[_IN, _KEY, _OUT]):
    """A transform that joins a source iterable with a second iterable by key.

    The right iterable is provided at construction time and materialized into
    an index keyed by ``on_key``. The left iterable is the source passed to
    ``__call__``. For best performance, place the smaller dataset on the right.

    Duplicate keys produce a cross-product of matching items, consistent with
    SQL join semantics.
    """

    def __init__(self, right: Iterable[_IN], *, on_key: Callable[[_IN], _KEY]):
        """Initialize the join transform.

        Args:
            right: The right iterable to join against. Materialized immediately.
            on_key: Key extractor for items of type _IN.
        """
        self._on = on_key

        # Materialize the right side into a keyed index.
        self._rindex: dict[_KEY, list[_IN]] = defaultdict(list)
        for item in right:
            self._rindex[self._on(item)].append(item)


class InnerJoinTransform(JoinTransform[_IN, _KEY, tuple[_IN, _IN]], Generic[_IN, _KEY]):
    """An inner join: yields only pairs where the key exists in both sides."""

    def __call__(self, src: Iterable[_IN]) -> Iterable[tuple[_IN, _IN]]:
        """Yield ``(left, right)`` for each left item whose key matches a right item."""
        yield from (
            (left, right)
            for left in src
            for right in self._rindex.get(self._on(left), [])
        )


class LeftJoinTransform(
    JoinTransform[_IN, _KEY, tuple[_IN, _IN | None]], Generic[_IN, _KEY]
):
    """A left join: yields every left item, paired with ``None`` when no right match exists."""

    def __call__(self, src: Iterable[_IN]) -> Iterable[tuple[_IN, _IN | None]]:
        """Yield ``(left, right)`` for each left item, substituting ``None`` for unmatched rights."""
        yield from (
            (left, right)
            for left in src
            for right in self._rindex.get(self._on(left), [None])
        )


class RightJoinTransform(
    JoinTransform[_IN, _KEY, tuple[_IN | None, _IN]], Generic[_IN, _KEY]
):
    """A right join: yields every right item, paired with ``None`` when no left match exists."""

    def __call__(self, src: Iterable[_IN]) -> Iterable[tuple[_IN | None, _IN]]:
        """Yield ``(left, right)`` for matched pairs, then ``(None, right)`` for unmatched rights."""
        lkeys: set[_KEY] = set()
        yield from (
            (left, right)
            for left in src
            for right in self._rindex.get(_add_key(lkeys, self._on(left)), [])
        )

        # Yield unmatched right items.
        yield from (
            (None, right)
            for rkey, rvals in self._rindex.items()
            if rkey not in lkeys
            for right in rvals
        )


class OuterJoinTransform(
    JoinTransform[_IN, _KEY, tuple[_IN | None, _IN | None]], Generic[_IN, _KEY]
):
    """A full outer join: yields all items from both sides, with ``None`` filling unmatched positions."""

    def __call__(self, src: Iterable[_IN]) -> Iterable[tuple[_IN | None, _IN | None]]:
        """Yield ``(left, right)`` for matches, ``(left, None)`` for unmatched lefts, and ``(None, right)`` for unmatched rights."""
        lkeys: set[_KEY] = set()
        yield from (
            (left, right)
            for left in src
            for right in self._rindex.get(_add_key(lkeys, self._on(left)), [None])
        )

        # Yield unmatched right items.
        yield from (
            (None, right)
            for rkey, rvals in self._rindex.items()
            if rkey not in lkeys
            for right in rvals
        )
