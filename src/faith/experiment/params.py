# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Types used to configure experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dataclasses_json import DataClassJsonMixin

from faith._internal.types.flags import GenerationMode, PathWithAnnotations, SampleRatio
from faith.benchmark.formatting.prompt import PromptFormatter


@dataclass
class DataSamplingParams(DataClassJsonMixin):
    """Parameters for dataset sampling."""

    sample_size: int | None = None


@dataclass
class ExperimentParams:
    """Parameters that define a set of benchmark experiments to run."""

    benchmark_names: Sequence[str] | None
    custom_benchmark_paths: Sequence[Path] | None
    generation_mode: GenerationMode
    prompt_format: PromptFormatter
    n_shot: Sequence[SampleRatio]
    model_paths: Sequence[PathWithAnnotations]
    num_trials: int
    initial_seed: int
