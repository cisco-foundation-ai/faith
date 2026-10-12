# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from faith._internal.algo.hash import dict_sha256
from faith._internal.io.resources import benchmarks_root
from faith._internal.io.store import Store
from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.benchmark.spec import BenchmarkSpec
from faith._types.config.agentic import AgenticEnvironmentConfig
from faith._types.model.generation import GenerationMode, GenParams
from faith._types.model.prompt import PromptFormatter
from faith.benchmark.config import load_config_from_path
from faith.experiment.experiment import BenchmarkExperiment


class FakeStore(Store):
    """A fake store implementation for testing purposes."""

    def __init__(self, root_path: Path):
        """Initialize the FakeStore with a root path."""
        self._root_path = root_path

    @property
    def path(self) -> Path:
        """The path of the datastore is just the root path for testing."""
        return self._root_path

    # pylint: disable-next=unused-argument
    def pull(self, *, raise_on_error: bool = False) -> Path:
        """Fake pull just returns the root path."""
        return self._root_path

    # pylint: disable-next=unused-argument
    def push(self, *, raise_on_error: bool = False) -> None:
        """Fake push does nothing."""

    def sub_store(self, sub_path: Path) -> Store:
        """Return a new FakeStore rooted at the given sub_path."""
        return FakeStore(self._root_path / sub_path)


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
        generation_mode=GenerationMode.CHAT_COMP,
        prompt_format=PromptFormatter.CHAT,
        n_shot=SampleRatio(5),
        model_name="example_model",
        gen_params=gen_params,
        root_datastore=FakeStore(Path("/tmp")),
        num_trials=3,
        initial_seed=3,
    )
    assert experiment.benchmark_config.metadata.name == "for-unit-test-only"
    assert experiment.datastore.path.parent == Path(
        "/tmp/for-unit-test-only/example_model/chat/chat_comp/5_shot"
    )
    assert experiment.datastore.path.name.startswith("gen_params_")
    assert experiment.benchmark_spec == BenchmarkSpec(
        name="for-unit-test-only",
        generation_mode=GenerationMode.CHAT_COMP,
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
            root_datastore=FakeStore(Path("/tmp")),
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
            root_datastore=FakeStore(Path("/tmp")),
            num_trials=0,
            initial_seed=27,
        )


def _agentic_experiment(
    benchmark_dir: str, environment_config: AgenticEnvironmentConfig | None = None
) -> BenchmarkExperiment:
    return BenchmarkExperiment(
        benchmark_path=Path(__file__).parent / "testdata" / benchmark_dir,
        generation_mode=GenerationMode.CHAT_COMP,
        prompt_format=PromptFormatter.CHAT,
        n_shot=SampleRatio(0),
        model_name="example_model",
        gen_params=GenParams(
            temperature=0.1, top_p=0.5, max_completion_tokens=100, kwargs={}
        ),
        root_datastore=FakeStore(Path("/tmp")),
        num_trials=1,
        initial_seed=0,
        environment_config=environment_config,
    )


def test_agentic_config_forks_the_experiment_path() -> None:
    path = _agentic_experiment("bench-agentic").datastore.path
    assert path.parent.parent == Path(
        "/tmp/bench-agentic/example_model/chat/chat_comp/0_shot"
    )
    assert path.parent.name.startswith("gen_params_")
    assert re.fullmatch(r"agentic_[0-9a-f]{16}", path.name)

    raised_cap = _agentic_experiment("bench-agentic-limits").datastore.path
    assert raised_cap.parent == path.parent
    assert raised_cap.name != path.name


@pytest.mark.parametrize(
    "benchmark_dir",
    ["bench-agentic-described", "bench-agentic-verbose"],
    ids=[
        "an-edit-outside-the-agentic-config",
        "the-same-config-with-its-defaults-spelled-out",
    ],
)
def test_the_agentic_segment_forks_on_nothing_else(benchmark_dir: str) -> None:
    """Neither a non-agentic edit nor an equivalent authoring may move the path."""
    assert (
        _agentic_experiment(benchmark_dir).datastore.path
        == _agentic_experiment("bench-agentic").datastore.path
    )


def test_a_deployment_override_does_not_fork_the_experiment_path() -> None:
    """A run's backend is where it ran, not what it answered, so the path is shared."""
    docker = _agentic_experiment("bench-agentic", AgenticEnvironmentConfig("docker"))
    k8s = _agentic_experiment("bench-agentic", AgenticEnvironmentConfig("kubernetes"))

    # The override reaches the config the experiment records...
    assert docker.benchmark_config.agentic_config is not None
    assert k8s.benchmark_config.agentic_config is not None
    assert (
        docker.benchmark_config.agentic_config.environment.type,
        k8s.benchmark_config.agentic_config.environment.type,
    ) == ("docker", "kubernetes")
    # ...but both are the same experiment, so they share one datastore.
    assert docker.datastore.path == k8s.datastore.path
    assert k8s.datastore.path == _agentic_experiment("bench-agentic").datastore.path


def test_secret_redaction_does_not_change_the_authored_cache_path() -> None:
    """Recording must not collapse different secrets or orphan pre-fix checkpoints."""
    base = load_config_from_path(Path(__file__).parent / "testdata/bench-agentic")
    assert base.agentic_config is not None
    paths = []
    for secret in ["fake-first", "fake-second"]:
        raw = {"environment": {"API_KEY": secret}}
        agentic = replace(
            base.agentic_config, environment=AgenticEnvironmentConfig("docker", raw)
        )
        # This is the shape to_dict returned before environment redaction existed.
        original = agentic.to_dict()
        original["environment"]["config"] = raw
        with patch(
            "faith.experiment.experiment.load_config_from_path",
            return_value=replace(base, agentic_config=agentic),
        ):
            path = _agentic_experiment("bench-agentic").datastore.path
        assert path.name == f"agentic_{dict_sha256(original)[-16:]}"
        paths.append(path)
    assert paths[0] != paths[1]


def test_benchmark_experiment_iteration() -> None:
    gen_params = GenParams(
        temperature=0.1,
        top_p=0.5,
        max_completion_tokens=100,
        kwargs={},
    )
    experiment = BenchmarkExperiment(
        benchmark_path=benchmarks_root() / "for-unit-test-only",
        generation_mode=GenerationMode.CHAT_COMP,
        prompt_format=PromptFormatter.CHAT,
        n_shot=SampleRatio(5),
        model_name="example_model",
        gen_params=gen_params,
        root_datastore=FakeStore(Path("/tmp")),
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
