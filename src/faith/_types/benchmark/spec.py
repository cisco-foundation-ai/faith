# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides the BenchmarkSpec type for specifying a benchmark."""

from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin, config

from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.model.generation import GenerationMode
from faith._types.model.prompt import PromptFormatter


@dataclass(frozen=True)
class BenchmarkSpec(DataClassJsonMixin):
    """Specification for a benchmark."""

    name: str
    generation_mode: GenerationMode = field(
        metadata=config(decoder=GenerationMode, encoder=str)
    )
    prompt_format: PromptFormatter = field(
        metadata=config(decoder=PromptFormatter, encoder=str)
    )
    n_shot: SampleRatio = field(
        metadata=config(decoder=SampleRatio.from_string, encoder=str)
    )
