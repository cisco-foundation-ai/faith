# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from faith._internal.algo.hash import dict_sha256


@pytest.mark.parametrize(
    "input_dict, expected_hash",
    [
        (
            {"a": 1, "b": 2},
            "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777",
        ),
        (
            {"b": 2, "a": [1, 2, 9]},
            "2c3d3f804a553fd8715611fb6c9526cb3bc8f6f7cb05117d9ec94220abee1f21",
        ),
        (
            {"a": [1, 2, 9], "b": 2},
            "2c3d3f804a553fd8715611fb6c9526cb3bc8f6f7cb05117d9ec94220abee1f21",
        ),
        (
            {"a": 1, "b": {"d": 0.5, "c": "foo"}},
            "e9ebbec7fd752885e827264f60679b66bab6864933872435f134c5ff9e44d202",
        ),
        (
            {"b": {"d": 0.5, "c": "foo"}, "a": 1},
            "e9ebbec7fd752885e827264f60679b66bab6864933872435f134c5ff9e44d202",
        ),
        (
            {"b": {"c": "foo", "d": 0.5}, "a": 1},
            "e9ebbec7fd752885e827264f60679b66bab6864933872435f134c5ff9e44d202",
        ),
    ],
)
def test_dict_sha256(input_dict: dict[str, Any], expected_hash: str) -> None:
    """Test the dict_sha256 function."""
    assert dict_sha256(input_dict) == expected_hash
