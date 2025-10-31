# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for hashing dictionaries."""

import hashlib
from typing import Any

import orjson


def dict_sha256(d: dict[Any, Any]) -> str:
    """Compute the SHA-256 hash of a dictionary."""
    json_bytes = orjson.dumps(
        d,
        option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
    )
    return hashlib.sha256(json_bytes).hexdigest()
