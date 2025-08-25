# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._internal.io.benchmarks import benchmarks_root


def test_benchmarks_root() -> None:
    """Test that the benchmarks root directory exists and is a directory."""
    assert benchmarks_root().exists(), "Benchmarks root does not exist."
    assert benchmarks_root().is_dir(), "Benchmarks root is not a directory."
    assert benchmarks_root().is_absolute(), "Benchmarks root is not an absolute path."
    assert str(benchmarks_root()).endswith("faith/__benchmarks__")
