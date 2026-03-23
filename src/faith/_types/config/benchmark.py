# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Top-level benchmark configuration type."""

from dataclasses import dataclass, field
from enum import auto

from dataclasses_json import DataClassJsonMixin, config

from faith._types.config.format import FormatConfig
from faith._types.config.metadata import MetadataConfig
from faith._types.config.scoring import OutputProcessingConfig
from faith._types.config.source import SourceConfig
from faith._types.enums import CIStrEnum


class ShortAnswerType(CIStrEnum):
    """Enum for validation types for short answer benchmarks."""

    # Short answer benchmarks where each answer is treated as a set of labels.
    LABEL_SET = auto()
    # Short answer benchmarks where each answer is treated as a single string label.
    STRING_MATCH = auto()
    # Short answer benchmarks where each answer is scored by domain-specific scores.
    DOMAIN_SPECIFIC = auto()


class LongAnswerType(CIStrEnum):
    """Enum for validation types for long answer benchmarks."""

    # Long answer benchmarks where each answer is free-form text
    # to be evaluated by an LLM.
    FREE_FORM = auto()


@dataclass(frozen=True)
class MCQAConfig(DataClassJsonMixin):
    """Configuration for multiple choice question-answering benchmarks."""

    answer_symbols: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])


@dataclass(frozen=True)
class SAQAConfig(DataClassJsonMixin):
    """Configuration for short answer question-answering benchmarks."""

    type: ShortAnswerType = field(metadata=config(encoder=str, decoder=ShortAnswerType))


@dataclass(frozen=True)
class LAQAConfig(DataClassJsonMixin):
    """Configuration for long answer question-answering benchmarks."""

    type: LongAnswerType = field(metadata=config(encoder=str, decoder=LongAnswerType))


@dataclass(frozen=True)
class BenchmarkConfig(DataClassJsonMixin):
    """Top-level benchmark configuration loaded from benchmark.yaml files."""

    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    format: FormatConfig = field(default_factory=FormatConfig)
    output_processing: OutputProcessingConfig = field(
        default_factory=OutputProcessingConfig
    )
    mcqa_config: MCQAConfig | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    saqa_config: SAQAConfig | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    laqa_config: LAQAConfig | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )

    def __post_init__(self) -> None:
        if (
            sum(
                c is not None
                for c in [self.mcqa_config, self.saqa_config, self.laqa_config]
            )
            > 1
        ):
            raise ValueError(
                "At most one of mcqa_config, saqa_config, laqa_config may be provided."
            )
