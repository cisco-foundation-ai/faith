# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for creating and managing paths."""

import re


def canonical_segment(segment: str) -> str:
    """Return a canonical segment of a path, ensuring it is a valid identifier.

    Args:
        segment: The segment to canonicalize.

    Returns:
        A canonicalized segment that is safe for use in paths.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", segment)
