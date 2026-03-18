# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Top-level benchmark configuration type."""

from dataclasses import dataclass, field
from enum import auto
from typing import Any

from dacite import Config, from_dict

from faith._types.configs.format import FormatConfig
from faith._types.configs.metadata import BenchmarkState, MetadataConfig
from faith._types.configs.patterns import AnswerFormat, Disambiguation
from faith._types.configs.scoring import OutputProcessingConfig, ScoreFnConfig
from faith._types.configs.source import DataFileType, SourceConfig
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
class MCQAConfig:
    """Configuration for multiple choice question-answering benchmarks."""

    answer_symbols: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])


@dataclass(frozen=True)
class SAQAConfig:
    """Configuration for short answer question-answering benchmarks."""

    type: ShortAnswerType


@dataclass(frozen=True)
class LAQAConfig:
    """Configuration for long answer question-answering benchmarks."""

    type: LongAnswerType


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level benchmark configuration loaded from benchmark.yaml files."""

    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    format: FormatConfig = field(default_factory=FormatConfig)
    output_processing: OutputProcessingConfig = field(
        default_factory=OutputProcessingConfig
    )
    mcqa_config: MCQAConfig | None = None
    saqa_config: SAQAConfig | None = None
    laqa_config: LAQAConfig | None = None

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

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "BenchmarkConfig":
        """Load a BenchmarkConfig from a raw dictionary (e.g. parsed YAML)."""

        def _score_fn_hook(d: dict[str, Any] | ScoreFnConfig) -> ScoreFnConfig:
            if isinstance(d, ScoreFnConfig):
                return d
            return ScoreFnConfig(
                type=d["type"],
                kwargs={k: v for k, v in d.items() if k != "type"},
            )

        return from_dict(
            data_class=BenchmarkConfig,
            data=data,
            config=Config(
                type_hooks={
                    BenchmarkState: BenchmarkState,
                    ShortAnswerType: ShortAnswerType,
                    LongAnswerType: LongAnswerType,
                    DataFileType: DataFileType,
                    AnswerFormat: AnswerFormat,
                    Disambiguation: Disambiguation,
                    ScoreFnConfig: _score_fn_hook,
                },
            ),
        )
