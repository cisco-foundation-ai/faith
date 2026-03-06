# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._internal.io.resources import benchmarks_root, models_root


def test_benchmarks_root() -> None:
    """Test that the benchmarks root directory exists and is a directory."""
    assert benchmarks_root().exists(), "Benchmarks root does not exist."
    assert benchmarks_root().is_dir(), "Benchmarks root is not a directory."
    assert benchmarks_root().is_absolute(), "Benchmarks root is not an absolute path."
    assert str(benchmarks_root()).endswith("faith/__benchmarks__")


def test_models_root() -> None:
    """Test that the models root directory exists and is a directory."""
    assert models_root().exists(), "Models root does not exist."
    assert models_root().is_dir(), "Models root is not a directory."
    assert models_root().is_absolute(), "Models root is not an absolute path."
    assert str(models_root()).endswith("faith/__models__")
