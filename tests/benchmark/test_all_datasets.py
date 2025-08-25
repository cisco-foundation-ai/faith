# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from faith._internal.algo.hash import dict_sha256
from faith._internal.io.benchmarks import benchmarks_root
from faith._internal.types.flags import GenerationMode, SampleRatio
from faith.benchmark.benchmark import Benchmark
from faith.benchmark.config import load_config_from_name
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.benchmark.load import load_benchmark
from faith.benchmark.types import BenchmarkSpec


def load_benchmark_for_test(name: str) -> Benchmark:
    """Load a benchmark with a standard spec for coarse testing purposes."""
    benchmark_spec = BenchmarkSpec(
        name=name,
        generation_mode=GenerationMode.CHAT_COMPLETION,
        prompt_format=PromptFormatter.CHAT,
        n_shot=SampleRatio(0),
    )
    benchmark_config = load_config_from_name(name)
    return load_benchmark(
        benchmark_spec,
        benchmark_config,
        path=benchmarks_root() / name,
        seed=12345678910987654321,
    )


def hash_dataframe(df: pd.DataFrame) -> str:
    """Generate a hash for a DataFrame."""
    return dict_sha256(df.to_dict(orient="list"))


@pytest.mark.slow
def test_ctibench_ate() -> None:
    """Test the CTIbench ATE benchmark."""
    benchmark = load_benchmark_for_test("ctibench-ate")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 60

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "4a48ef39097a067e7e7ef5b92b32e74d5249b747634baa1104a25bb4c1773461"
    )


@pytest.mark.slow
def test_ctibench_mcqa() -> None:
    """Test the CTIbench MCQA benchmark."""
    benchmark = load_benchmark_for_test("ctibench-mcqa")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 2500

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "94aaa09fbafceac467791c3ce5d0101c77f1dca848abafa0e6ca1a3548bb2c37"
    )


@pytest.mark.slow
def test_ctibench_rcm() -> None:
    """Test the CTIbench RCM benchmark."""
    benchmark = load_benchmark_for_test("ctibench-rcm")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 1000

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "cc894426cb94e83873f8fc330c079125d8785ad93cf70608b6ba62274c516ecd"
    )


@pytest.mark.slow
def test_ctibench_taa() -> None:
    """Test the CTIbench TAA benchmark."""
    benchmark = load_benchmark_for_test("ctibench-taa")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 50

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "9c952db0ea68f1f6a26d6ae78b345fc679f8052cee0f08c769ce8edd885efdfc"
    )


@pytest.mark.slow
def test_ctibench_vsp() -> None:
    """Test the CTIbench VSP benchmark."""
    benchmark = load_benchmark_for_test("ctibench-vsp")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 1000

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "b8839d6188c57523267b42404c7e2115a0300b91f5fe813d6338edcceab941de"
    )


@pytest.mark.slow
def test_cybermetric_80() -> None:
    """Test the Cybermetric-80 benchmark."""
    benchmark = load_benchmark_for_test("cybermetric-80")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 80

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "3aadbc09ac1a840d7137c24b061fc048a3a78c468b29091dc5b2bdc74c3d1338"
    )


@pytest.mark.slow
def test_cybermetric_500() -> None:
    """Test the Cybermetric-500 benchmark."""
    benchmark = load_benchmark_for_test("cybermetric-500")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 500

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "2797644b3e0fb567110cb1e0e5a54faa6ca719eda5879f05c98d127ec50401c4"
    )


@pytest.mark.slow
def test_cybermetric_2000() -> None:
    """Test the Cybermetric-2000 benchmark."""
    benchmark = load_benchmark_for_test("cybermetric-2000")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 2000

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "d0a9955e157486c6ac4eeada507f970d435d8410b2a3041427d85b359e2aaca1"
    )


@pytest.mark.slow
def test_cybermetric_10000() -> None:
    """Test the Cybermetric-10000 benchmark."""
    benchmark = load_benchmark_for_test("cybermetric-10000")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 10180

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "687c3f57f00fcfb73f856bd3659d98f5aebfa8a3816b5a3c0c3dd9cac4eec077"
    )


@pytest.mark.slow
def test_mmlu_all() -> None:
    """Test the MMLU-all benchmark."""
    benchmark = load_benchmark_for_test("mmlu-all")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 14042

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "403f569ac6e88884964c53339f0a613c6f2c6e0e04d99001110767143291b800"
    )


@pytest.mark.slow
def test_mmlu_security() -> None:
    """Test the MMLU-security benchmark."""
    benchmark = load_benchmark_for_test("mmlu-security")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 100

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "72811d65b24d0aab59a0c0ca4a53c1c8fd4fb97860f6b967d0dc97dd4a061d47"
    )


@pytest.mark.slow
def test_secbench_mcqa_eng() -> None:
    """Test the SecBench MCQA English benchmark."""
    benchmark = load_benchmark_for_test("secbench-mcqa-eng")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 595

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "31be71026da26cd4559e76ea043f8922d634427f5f3e26e3ce350f1bf56d7fb9"
    )


@pytest.mark.slow
def test_secbench_mcqa_eng_reasoning() -> None:
    """Test the SecBench MCQA English Reasoning benchmark."""
    benchmark = load_benchmark_for_test("secbench-mcqa-eng-reasoning")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 43

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "d91b06a8f4713810f0ce0c68d8f8e52944a7d2b03d56474fb107b5ddf01a9de2"
    )


@pytest.mark.slow
def test_seceval() -> None:
    """Test the SecEval benchmark."""
    benchmark = load_benchmark_for_test("seceval")
    dataset = benchmark.build_dataset()
    assert len([_ for _ in dataset.iter_data()]) == 1255

    # The hash acts as a watermark for the benchmark data to ensure it hasn't changed.
    assert (
        hash_dataframe(dataset._benchmark_data)
        == "2e3b74cba5069bb8a2a1b2e70d2a27e4aeb97999b758ade35d4dac715f3f9f8d"
    )
