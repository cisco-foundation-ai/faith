# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

from pytest_unordered import unordered

from faith._internal.io.resources import benchmarks_root
from faith.benchmark.config import load_config_from_path
from faith.benchmark.listing import (
    benchmark_choices,
    benchmark_groups,
    choices_to_benchmarks,
    find_benchmarks,
)


def test_benchmark_choices() -> None:
    assert benchmark_choices() == [
        "all",
        "all-reasoning",
        "all-security",
        "cti-reasoning",
        "!cti-reasoning",
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
        "cwe-prediction",
        "!cwe-prediction",
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
            "cti-reasoning",
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cwe-prediction",
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
            "cti-reasoning",
            "secbench-mcqa-eng-reasoning",
        ]
    )
    assert choices_to_benchmarks(["all-security"]) == benchmark_subpath(
        [
            "cti-reasoning",
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cwe-prediction",
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
            "cti-reasoning",
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cwe-prediction",
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
            "cti-reasoning",
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cwe-prediction",
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
    # Test that all of the built-in benchmarks are found.
    assert find_benchmarks(benchmarks_root()) == unordered(
        benchmark_subpath(
            [
                "cti-reasoning",
                "ctibench-ate",
                "ctibench-mcqa",
                "ctibench-rcm",
                "ctibench-vsp",
                "cwe-prediction",
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

    # Test that the correct external benchmarks are found. find_benchmarks is
    # unfiltered by type, so an enabled agentic benchmark is still listed here
    # (i.e. individually selectable) even though it is excluded from the groups.
    assert find_benchmarks(Path(__file__).parent / "testdata") == unordered(
        [
            Path(__file__).parent / "testdata" / "bench-a",
            Path(__file__).parent / "testdata" / "bench-agentic",
            Path(__file__).parent / "testdata" / "sub" / "bench-c",
        ]
    )


def test_every_group_excludes_agentic_benchmarks() -> None:
    """No group sweeps in an agentic benchmark, but it stays selectable by name.

    Covers every group rather than a sampled one, so a group added later without
    the exclusion fails here.
    """
    testdata = Path(__file__).parent / "testdata"
    # bench-agentic is categorized security + reasoning, so it would land in
    # every group if it were not excluded.
    with patch("faith.benchmark.listing.benchmarks_root", return_value=testdata):
        for group in benchmark_groups():
            selected = choices_to_benchmarks([group])
            assert testdata / "bench-agentic" not in selected, group

        assert choices_to_benchmarks(["all"]) == [
            testdata / "bench-a",
            testdata / "sub" / "bench-c",
        ]
        assert choices_to_benchmarks(["bench-agentic"]) == [testdata / "bench-agentic"]


def test_all_benchmarks_have_primary_metric() -> None:
    """Ensure all benchmarks define a primary_metric for analytics."""
    core_benchmarks = find_benchmarks(benchmarks_root())
    for benchmark_path in core_benchmarks:
        config = load_config_from_path(benchmark_path)
        primary_metric = config.output_processing.primary_metric

        assert (
            primary_metric and len(primary_metric) > 0
        ), f"Benchmark {benchmark_path.name} must have a non-empty 'primary_metric'."
