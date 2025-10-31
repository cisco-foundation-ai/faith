# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides `benchmarks_root` function to get the path to packaged benchmarks."""

from importlib import resources
from pathlib import Path


def benchmarks_root() -> Path:
    """Return the root path of the benchmarks directory."""
    with resources.as_file(
        resources.files("faith") / "__benchmarks__"
    ) as benchmarks_path:
        return benchmarks_path
