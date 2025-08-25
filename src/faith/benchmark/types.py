# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""This module defines the common schema for the QA examples used in benchmarks."""
from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin, config

from faith._internal.algo.hash import dict_sha256
from faith._internal.types.flags import GenerationMode, SampleRatio
from faith.benchmark.formatting.prompt import PromptFormatter


@dataclass(frozen=True)
class BenchmarkSpec(DataClassJsonMixin):
    """Specification for a benchmark."""

    name: str
    generation_mode: GenerationMode = field(
        metadata=config(decoder=GenerationMode, encoder=str)
    )
    prompt_format: PromptFormatter = field(
        metadata=config(decoder=PromptFormatter.from_string, encoder=str)
    )
    n_shot: SampleRatio = field(
        metadata=config(decoder=SampleRatio.from_string, encoder=str)
    )

    def sha256(self) -> str:
        """Compute the SHA-256 hash of this example."""
        return dict_sha256(self.to_dict())
