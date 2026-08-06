# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Shared score value type used by scoring functions and graders."""

from typing import NotRequired, TypedDict


class Score(TypedDict):
    """A score value and any associated metadata for a single metric.

    Attributes:
        value: The score rescaled into the scoring function's configured range.
        raw_value: The score in the scoring function's own raw range, where
            higher indicates a better match.
    """

    value: NotRequired[float]
    raw_value: float
