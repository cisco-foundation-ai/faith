# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from faith._internal.io.benchmarks import benchmarks_root
from faith._internal.types.flags import GenerationMode, SampleRatio
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.benchmark.types import BenchmarkSpec
from faith.experiment.experiment import BenchmarkExperiment
from faith.model.params import GenParams


def test_benchmark_experiment() -> None:
    gen_params = GenParams(
        temperature=0.1,
        top_p=0.5,
        max_completion_tokens=100,
        kwargs={},
    )

    # Test initialization with valid parameters
    experiment = BenchmarkExperiment(
        benchmark_path=benchmarks_root() / "for-unit-test-only",
        generation_mode=GenerationMode.CHAT_COMPLETION,
        prompt_format=PromptFormatter.CHAT,
        n_shot=SampleRatio(5),
        model_name="example_model",
        gen_params=gen_params,
        datastore_path=Path("/tmp"),
        num_trials=3,
        initial_seed=3,
    )
    assert experiment.benchmark_config["metadata"]["name"] == "for-unit-test-only"
    assert str(experiment.experiment_dir).startswith(
        "/tmp/for-unit-test-only/example_model/chat/chat_comp/5_shot/gen_params_"
    )
    assert experiment.benchmark_spec == BenchmarkSpec(
        name="for-unit-test-only",
        generation_mode=GenerationMode.CHAT_COMPLETION,
        prompt_format=PromptFormatter.CHAT,
        n_shot=SampleRatio(5),
    )
    assert len(experiment) == 3

    with pytest.raises(
        AssertionError,
        match="Benchmark path '.*/does-not-exist' is not an existing directory.",
    ):
        # Test initialization with invalid benchmark name.
        BenchmarkExperiment(
            benchmark_path=benchmarks_root() / "does-not-exist",
            generation_mode=GenerationMode.LOGITS,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(0),
            model_name="example_model",
            gen_params=gen_params,
            datastore_path=Path("/tmp"),
            num_trials=3,
            initial_seed=81,
        )
    with pytest.raises(
        AssertionError, match="Number of trials must be positive, but got 0."
    ):
        # Test initialization with invalid number of trials.
        BenchmarkExperiment(
            benchmark_path=benchmarks_root() / "for-unit-test-only",
            generation_mode=GenerationMode.NEXT_TOKEN,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(1, 4),
            model_name="example_model",
            gen_params=gen_params,
            datastore_path=Path("/tmp"),
            num_trials=0,
            initial_seed=27,
        )


def test_experiment_iteration() -> None:
    gen_params = GenParams(
        temperature=0.1,
        top_p=0.5,
        max_completion_tokens=100,
        kwargs={},
    )
    experiment = BenchmarkExperiment(
        benchmark_path=benchmarks_root() / "for-unit-test-only",
        generation_mode=GenerationMode.CHAT_COMPLETION,
        prompt_format=PromptFormatter.CHAT,
        n_shot=SampleRatio(5),
        model_name="example_model",
        gen_params=gen_params,
        datastore_path=Path("/tmp"),
        num_trials=2,
        initial_seed=375,
    )

    # Test __iter__ and __next__ by iterating through the experiment
    trial_count = 0
    for i, (_, trial_path) in enumerate(experiment):
        trial_count += 1
        assert trial_path.parts[0] == "trials"
        assert trial_path.name == "benchmark-log.json"
        assert f"{375 + i}" in str(trial_path)
    assert trial_count == 2
