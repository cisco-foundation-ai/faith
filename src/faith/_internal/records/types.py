# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum, auto
from typing import Any, TypeAlias

# Defines a record as a dictionary with string keys and any type of values.
Record: TypeAlias = dict[str, Any]


class RecordStatus(StrEnum):
    """Indicates whether a record is clean (unchanged) or dirty (new or updated)."""

    CLEAN = auto()
    DIRTY = auto()
