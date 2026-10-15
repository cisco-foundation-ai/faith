# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Types used to configure experiments."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import DataClassJsonMixin

from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.config.agentic import AgenticEnvironmentConfig
from faith._types.model.generation import GenerationMode


@dataclass(frozen=True)
class DataSamplingParams(DataClassJsonMixin):
    """Parameters for dataset sampling."""

    sample_size: int | None = None


@dataclass(frozen=True)
class ExperimentParams:
    """Parameters that define a set of benchmark experiments to run.

    Attributes:
        benchmark_names: Names of the packaged benchmarks to run.
        custom_benchmark_paths: Paths to out-of-tree benchmarks to run as well.
        generation_mode: How the model is queried (agentic benchmarks pin their own).
        n_shot: In-context example counts to fan out over (agentic pins 0).
        num_trials: Trials per benchmark.
        initial_seed: Seed of the first trial; later trials increment it.
        environment_config: A deployment's agentic environment override, or None to
            use each benchmark's own. Applies to every benchmark in the run: one
            run, one place the agents execute.
    """

    benchmark_names: Sequence[str] | None
    custom_benchmark_paths: Sequence[Path] | None
    generation_mode: GenerationMode
    n_shot: Sequence[SampleRatio]
    num_trials: int
    initial_seed: int
    environment_config: AgenticEnvironmentConfig | None = None
