# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Functions that list benchmarks with their states and categories."""

from enum import Enum
from pathlib import Path
from typing import Sequence

from faith._internal.io.benchmarks import benchmarks_root
from faith.benchmark.config import load_config_from_path


class BenchmarkState(Enum):
    """Enum for benchmark states."""

    ENABLED = "enabled"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"
    TEST_ONLY = "test_only"

    @staticmethod
    def from_string(s: str) -> "BenchmarkState":
        """Convert a string to a BenchmarkState enum."""
        try:
            return BenchmarkState[s.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown benchmark state: {s}") from e


def _get_benchmark_paths(
    root_dir: Path,
    allowed_states: set[BenchmarkState],
    allowed_categories: set[str] | None = None,
) -> Sequence[str]:
    """Get a list of all sub-paths of `root_dir` that are valid benchmark paths."""
    return sorted(
        [
            str(f.parent.relative_to(root_dir))
            for f in root_dir.glob("**/benchmark.yaml")
            if (metadata := load_config_from_path(f.parent)["metadata"])
            and BenchmarkState.from_string(metadata.get("state", "enabled"))
            in allowed_states
            and (
                not allowed_categories
                or allowed_categories.intersection(metadata.get("categories", []))
            )
        ]
    )


def benchmark_choices() -> Sequence[str]:
    """Get a list of available benchmarks."""
    return ["all", "all-reasoning", "all-security"] + [
        opt
        for p in _get_benchmark_paths(
            benchmarks_root(),
            {BenchmarkState.ENABLED, BenchmarkState.EXPERIMENTAL},
        )
        for opt in [p, f"!{p}"]
    ]


def choices_to_benchmarks(choices: Sequence[str]) -> Sequence[Path]:
    """Convert a list of benchmark choices to benchmark names."""
    if not choices:
        return []
    group_name_mapping = {
        "all": _get_benchmark_paths(benchmarks_root(), {BenchmarkState.ENABLED}),
        "all-reasoning": _get_benchmark_paths(
            benchmarks_root(),
            {BenchmarkState.ENABLED},
            allowed_categories={"reasoning"},
        ),
        "all-security": _get_benchmark_paths(
            benchmarks_root(), {BenchmarkState.ENABLED}, allowed_categories={"security"}
        ),
    }
    unique_choices = list(
        dict.fromkeys(
            [
                c
                for group_name in choices
                for c in (group_name_mapping.get(group_name, [group_name]))
            ]
        ).keys()
    )
    positive_selections = [c for c in unique_choices if not c.startswith("!")]
    negative_selections = {c[1:] for c in unique_choices if c.startswith("!")}
    return [
        benchmarks_root() / c
        for c in positive_selections
        if c not in negative_selections
    ]


def find_benchmarks(root_dir: Path) -> Sequence[Path]:
    """Find all valid benchmarks in the `root_dir` directory."""
    return [
        root_dir / sub_path
        for sub_path in _get_benchmark_paths(root_dir, {BenchmarkState.ENABLED})
    ]
