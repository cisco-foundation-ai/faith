# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from faith._internal.io.benchmarks import benchmarks_root
from faith.benchmark.listing import (
    BenchmarkState,
    benchmark_choices,
    choices_to_benchmarks,
    find_benchmarks,
)


def test_benchmark_state_from_string() -> None:
    assert BenchmarkState.from_string("enabled") == BenchmarkState.ENABLED
    assert BenchmarkState.from_string("EXPERIMENTAL") == BenchmarkState.EXPERIMENTAL
    assert BenchmarkState.from_string("disabled") == BenchmarkState.DISABLED
    assert BenchmarkState.from_string("test_only") == BenchmarkState.TEST_ONLY

    with pytest.raises(ValueError):
        BenchmarkState.from_string("unknown")


def test_benchmark_choices() -> None:
    assert benchmark_choices() == [
        "all",
        "all-reasoning",
        "all-security",
        "ctibench-ate",
        "!ctibench-ate",
        "ctibench-mcqa",
        "!ctibench-mcqa",
        "ctibench-rcm",
        "!ctibench-rcm",
        "ctibench-taa",
        "!ctibench-taa",
        "ctibench-vsp",
        "!ctibench-vsp",
        "cybermetric-10000",
        "!cybermetric-10000",
        "cybermetric-2000",
        "!cybermetric-2000",
        "cybermetric-500",
        "!cybermetric-500",
        "cybermetric-80",
        "!cybermetric-80",
        "mmlu-all",
        "!mmlu-all",
        "mmlu-security",
        "!mmlu-security",
        "secbench-mcqa-eng",
        "!secbench-mcqa-eng",
        "secbench-mcqa-eng-reasoning",
        "!secbench-mcqa-eng-reasoning",
        "seceval",
        "!seceval",
    ]


def benchmark_subpath(sub_paths: list[str]) -> list[Path]:
    """Helper to convert sub-path strings to full benchmark paths."""
    return [benchmarks_root() / p for p in sub_paths]


def test_choices_to_benchmarks() -> None:
    assert choices_to_benchmarks([]) == []
    assert choices_to_benchmarks(["all"]) == benchmark_subpath(
        [
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cybermetric-10000",
            "cybermetric-2000",
            "cybermetric-500",
            "cybermetric-80",
            "mmlu-all",
            "mmlu-security",
            "secbench-mcqa-eng",
            "secbench-mcqa-eng-reasoning",
            "seceval",
        ]
    )
    assert choices_to_benchmarks(["all-reasoning"]) == benchmark_subpath(
        [
            "secbench-mcqa-eng-reasoning",
        ]
    )
    assert choices_to_benchmarks(["all-security"]) == benchmark_subpath(
        [
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cybermetric-10000",
            "cybermetric-2000",
            "cybermetric-500",
            "cybermetric-80",
            "mmlu-security",
            "secbench-mcqa-eng",
            "secbench-mcqa-eng-reasoning",
            "seceval",
        ]
    )
    assert choices_to_benchmarks(["mmlu-all", "all"]) == benchmark_subpath(
        [
            "mmlu-all",
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cybermetric-10000",
            "cybermetric-2000",
            "cybermetric-500",
            "cybermetric-80",
            "mmlu-security",
            "secbench-mcqa-eng",
            "secbench-mcqa-eng-reasoning",
            "seceval",
        ]
    )
    assert choices_to_benchmarks(["!mmlu-all", "all", "!seceval"]) == benchmark_subpath(
        [
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cybermetric-10000",
            "cybermetric-2000",
            "cybermetric-500",
            "cybermetric-80",
            "mmlu-security",
            "secbench-mcqa-eng",
            "secbench-mcqa-eng-reasoning",
        ]
    )
    assert choices_to_benchmarks(["mmlu-security"]) == benchmark_subpath(
        ["mmlu-security"]
    )


def test_find_benchmarks() -> None:
    core_benchmarks = find_benchmarks(benchmarks_root())
    assert set(core_benchmarks) == set(
        benchmark_subpath(
            [
                "ctibench-ate",
                "ctibench-mcqa",
                "ctibench-rcm",
                "ctibench-vsp",
                "cybermetric-10000",
                "cybermetric-2000",
                "cybermetric-500",
                "cybermetric-80",
                "mmlu-all",
                "mmlu-security",
                "secbench-mcqa-eng",
                "secbench-mcqa-eng-reasoning",
                "seceval",
            ]
        )
    )

    external_benchmarks = find_benchmarks(Path(__file__).parent / "testdata")
    assert set(external_benchmarks) == set(
        [
            Path(__file__).parent / "testdata" / "bench-a",
            Path(__file__).parent / "testdata" / "sub" / "bench-c",
        ]
    )
