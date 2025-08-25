# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._internal.io.paths import canonical_segment


def test_canonical_segment() -> None:
    """Test the canonical_segment function."""

    assert canonical_segment("") == ""
    assert canonical_segment("valid_segment") == "valid_segment"
    assert canonical_segment("1234") == "1234"
    assert canonical_segment("a-b_c.d") == "a-b_c.d"  # Dots and hyphens are allowed
    assert canonical_segment("invalid segment!") == "invalid_segment_"
    assert canonical_segment("another@invalid##segment") == "another_invalid__segment"
