# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Functions that list benchmarks with their states and categories."""

from collections.abc import Sequence
from pathlib import Path

from faith._internal.io.resources import benchmarks_root
from faith._types.config.metadata import BenchmarkState
from faith.benchmark.config import load_config_from_path


def _get_benchmark_paths(
    root_dir: Path,
    allowed_states: set[BenchmarkState],
    allowed_categories: set[str] | None = None,
    *,
    exclude_agentic: bool,
) -> Sequence[str]:
    """Get a list of all sub-paths of `root_dir` that are valid benchmark paths."""
    return sorted(
        [
            str(f.parent.relative_to(root_dir))
            for f in root_dir.glob("**/benchmark.yaml")
            if (config := load_config_from_path(f.parent))
            and config.metadata.state in allowed_states
            and (
                not allowed_categories
                or allowed_categories.intersection(config.metadata.categories)
            )
            and not (exclude_agentic and config.agentic_config is not None)
        ]
    )


# Group choice -> the categories it selects (None for every category).
_BENCHMARK_GROUPS: dict[str, set[str] | None] = {
    "all": None,
    "all-reasoning": {"reasoning"},
    "all-security": {"security"},
}


def benchmark_groups() -> Sequence[str]:
    """Names of the group choices that expand to several benchmarks."""
    return list(_BENCHMARK_GROUPS)


def benchmark_choices() -> Sequence[str]:
    """Get a list of available benchmarks."""
    return list(benchmark_groups()) + [
        opt
        for p in _get_benchmark_paths(
            benchmarks_root(),
            {BenchmarkState.ENABLED, BenchmarkState.EXPERIMENTAL},
            exclude_agentic=False,
        )
        for opt in [p, f"!{p}"]
    ]


def choices_to_benchmarks(choices: Sequence[str]) -> Sequence[Path]:
    """Convert a list of benchmark choices to benchmark names."""
    if not choices:
        return []
    # No group sweeps in an agentic benchmark: an agentic run is expensive (a
    # container plus many model calls per task), so it is opt-in by name. It
    # stays individually selectable via its own choice (see benchmark_choices,
    # which is unfiltered).
    group_name_mapping = {
        name: _get_benchmark_paths(
            benchmarks_root(),
            {BenchmarkState.ENABLED},
            allowed_categories=categories,
            exclude_agentic=True,
        )
        for name, categories in _BENCHMARK_GROUPS.items()
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
        for sub_path in _get_benchmark_paths(
            root_dir, {BenchmarkState.ENABLED}, exclude_agentic=False
        )
    ]
