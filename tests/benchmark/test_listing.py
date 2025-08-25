# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith.benchmark.listing import (
    BenchmarkState,
    benchmark_choices,
    choices_to_benchmarks,
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


def test_choices_to_benchmarks() -> None:
    assert choices_to_benchmarks([]) == []
    assert choices_to_benchmarks(["all"]) == [
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
    assert choices_to_benchmarks(["all-reasoning"]) == [
        "secbench-mcqa-eng-reasoning",
    ]
    assert choices_to_benchmarks(["all-security"]) == [
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
    assert choices_to_benchmarks(["mmlu-all", "all"]) == [
        "mmlu-all",
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
    assert choices_to_benchmarks(["!mmlu-all", "all", "!seceval"]) == [
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
    assert choices_to_benchmarks(["mmlu-security"]) == ["mmlu-security"]
