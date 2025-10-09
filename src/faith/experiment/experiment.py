# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Defines the `BenchmarkExperiment`, which manages an experiment for a benchmark."""

from pathlib import Path
from typing import Any, Iterator

from faith._internal.io.benchmarks import benchmarks_root
from faith._internal.types.flags import GenerationMode, PathWithAnnotations, SampleRatio
from faith.benchmark.benchmark import Benchmark
from faith.benchmark.config import load_config_from_path
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.benchmark.load import load_benchmark
from faith.benchmark.types import BenchmarkSpec
from faith.model.params import GenParams


class BenchmarkExperiment:
    """A class to manage an experiment for a benchmark over multiple trials.

    Experiments act as an iterator of `Benchmark` instances, each representing a trial
    of the benchmark with a specific seed. The experiment is configured with parameters
    benchmark name-or-path, generation mode, prompt format, number of shots, and any
    `kwargs` given. The datastore location and seed for the benchmark are changed for
    each trail, allowing for multiple runs over the same benchmark.
    """

    def __init__(
        self,
        name_or_path: str | PathWithAnnotations,
        generation_mode: GenerationMode,
        prompt_format: PromptFormatter,
        n_shot: SampleRatio,
        model_name: str,
        gen_params: GenParams,
        datastore_path: Path,
        num_trials: int,
        initial_seed: int,
        **kwargs: Any,
    ):
        """Initialize a benchmark experiment with the given experiment parameters."""
        assert (
            num_trials > 0
        ), f"Number of trials must be positive, but got {num_trials}."
        benchmark_name = (
            name_or_path
            if isinstance(name_or_path, str)
            else name_or_path.get_value("name")
        )
        assert (
            benchmark_name is not None
        ), f"A name must be provided as an annotation for custom benchmark '{str(name_or_path)}'."
        self._benchmark_dir = (
            benchmarks_root() / benchmark_name
            if isinstance(name_or_path, str)
            else name_or_path.path
        )
        assert (
            self._benchmark_dir.exists() and self._benchmark_dir.is_dir()
        ), f"Benchmark path '{self._benchmark_dir}' is not an existing directory."

        # State that specifies and configures the benchmark.
        self._benchmark_spec = BenchmarkSpec(
            name=benchmark_name,
            generation_mode=generation_mode,
            prompt_format=prompt_format,
            n_shot=n_shot,
        )
        self._benchmark_config = load_config_from_path(self._benchmark_dir)
        self._benchmark_kwargs = kwargs

        # State that specifes the model.
        self._model_name = model_name
        self._gen_params = gen_params

        # State for regulating the experiment and its trials.
        self._datastore_path = datastore_path
        self._num_trials = num_trials
        self._initial_seed = initial_seed
        self._trial = 0

    @property
    def benchmark_config(self) -> dict[str, Any]:
        """Returns the benchmark configuration loaded from its directory."""
        return self._benchmark_config

    @property
    def benchmark_spec(self) -> BenchmarkSpec:
        """Returns the specification of the benchmark for this experiment."""
        return self._benchmark_spec

    @property
    def experiment_dir(self) -> Path:
        """Returns the path of the datastore for this experiment."""
        return (
            self._datastore_path
            / self._benchmark_spec.name
            / self._model_name
            / str(self._benchmark_spec.prompt_format)
            / str(self._benchmark_spec.generation_mode)
            / f"{str(self._benchmark_spec.n_shot).replace('/', '_')}_shot"
            / f"gen_params_{self._gen_params.sha256()[-16:]}"
        )

    def __iter__(self) -> Iterator[tuple[Benchmark, Path]]:
        """Return an iterator over the benchmark trials."""
        return self

    def __len__(self) -> int:
        """Return the number of trials in the experiment."""
        return self._num_trials

    def __next__(self) -> tuple[Benchmark, Path]:
        """Create a new Benchmark instance for the next trial."""
        if self._trial >= self._num_trials:
            raise StopIteration
        trial_seed = self._initial_seed + self._trial
        benchmark = load_benchmark(
            self.benchmark_spec,
            self.benchmark_config,
            path=self._benchmark_dir,
            seed=trial_seed,
            **self._benchmark_kwargs,
        )
        trial_path = (
            Path("trials")
            / str(trial_seed)
            / self.benchmark_spec.sha256()[-16:]
            / "benchmark-log.json"
        )
        self._trial += 1
        return benchmark, trial_path
