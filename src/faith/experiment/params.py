# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Types used to configure experiments."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import DataClassJsonMixin

from faith._types.dataset.sample_ratio import SampleRatio
from faith._types.model.generation import GenerationMode


@dataclass(frozen=True)
class DataSamplingParams(DataClassJsonMixin):
    """Parameters for dataset sampling."""

    sample_size: int | None = None


@dataclass(frozen=True)
class ExperimentParams:
    """Parameters that define a set of benchmark experiments to run."""

    benchmark_names: Sequence[str] | None
    custom_benchmark_paths: Sequence[Path] | None
    generation_mode: GenerationMode
    n_shot: Sequence[SampleRatio]
    num_trials: int
    initial_seed: int
