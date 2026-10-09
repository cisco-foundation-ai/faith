# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides the paths to packaged benchmarks and models, and expansion of both."""

from importlib import resources
from pathlib import Path


def benchmarks_root() -> Path:
    """Return the root path of the benchmarks directory."""
    with resources.as_file(
        resources.files("faith") / "__benchmarks__"
    ) as benchmarks_path:
        return benchmarks_path


def models_root() -> Path:
    """Return the root path of the models directory."""
    with resources.as_file(resources.files("faith") / "__models__") as models_path:
        return models_path


def resolve_resource_path(path: str) -> Path:
    """Expand a leading `$BENCHMARKS_ROOT/` or `$MODELS_ROOT/` into a real path."""
    if path.startswith("$BENCHMARKS_ROOT/"):
        return benchmarks_root() / path.removeprefix("$BENCHMARKS_ROOT/")
    if path.startswith("$MODELS_ROOT/"):
        return models_root() / path.removeprefix("$MODELS_ROOT/")
    return Path(path)
