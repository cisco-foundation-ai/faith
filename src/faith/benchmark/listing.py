# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Functions that list benchmarks with their states and categories."""
from enum import Enum
from typing import Sequence

from faith._internal.io.benchmarks import benchmarks_root
from faith.benchmark.config import load_config_from_name


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
        except KeyError:
            raise ValueError(f"Unknown benchmark state: {s}")


def _benchmark_names(
    allowed_states: set[BenchmarkState], allowed_categories: set[str] | None = None
) -> Sequence[str]:
    """Get a list of all benchmark names that are in the allowed states."""
    return sorted(
        [
            name
            for f in benchmarks_root().glob("*/benchmark.yaml")
            if (name := f.parent.name)
            and (metadata := load_config_from_name(name)["metadata"])
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
        for n in _benchmark_names(
            allowed_states={BenchmarkState.ENABLED, BenchmarkState.EXPERIMENTAL},
        )
        for opt in [n, f"!{n}"]
    ]


def choices_to_benchmarks(choices: Sequence[str]) -> Sequence[str]:
    """Convert a list of benchmark choices to benchmark names."""
    if not choices:
        return []
    group_name_mapping = {
        "all": _benchmark_names(allowed_states={BenchmarkState.ENABLED}),
        "all-reasoning": _benchmark_names(
            allowed_states={BenchmarkState.ENABLED}, allowed_categories={"reasoning"}
        ),
        "all-security": _benchmark_names(
            allowed_states={BenchmarkState.ENABLED}, allowed_categories={"security"}
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
    negative_selections = set([c[1:] for c in unique_choices if c.startswith("!")])
    return [c for c in positive_selections if c not in negative_selections]
