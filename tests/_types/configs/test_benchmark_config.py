# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BenchmarkConfig.from_dict dacite loader."""

import pytest

from faith._types.configs.benchmark import (
    BenchmarkConfig,
    LAQAConfig,
    LongAnswerType,
    MCQAConfig,
    SAQAConfig,
    ShortAnswerType,
)
from faith._types.configs.format import FormatConfig, InstructionsConfig, PromptConfig
from faith._types.configs.metadata import BenchmarkState, MetadataConfig
from faith._types.configs.patterns import (
    AnswerFormat,
    CaptureTransform,
    Disambiguation,
    PatternDef,
)
from faith._types.configs.scoring import OutputProcessingConfig, ScoreFnConfig
from faith._types.configs.source import (
    HuggingFaceSourceConfig,
    SourceConfig,
    SourceOptionsConfig,
)


def test_metadata_with_state_enum() -> None:
    assert BenchmarkConfig.from_dict(
        {
            "metadata": {
                "name": "test-bench",
                "description": "A test benchmark.",
                "state": "experimental",
                "categories": ["security"],
                "urls": ["https://example.com"],
            }
        }
    ) == BenchmarkConfig(
        metadata=MetadataConfig(
            name="test-bench",
            description="A test benchmark.",
            state=BenchmarkState.EXPERIMENTAL,
            categories=["security"],
            urls=["https://example.com"],
        )
    )


def test_benchmark_types() -> None:
    assert BenchmarkConfig.from_dict(
        {"mcqa_config": {"answer_symbols": ["A", "B", "C"]}}
    ) == BenchmarkConfig(mcqa_config=MCQAConfig(answer_symbols=["A", "B", "C"]))
    assert BenchmarkConfig.from_dict(
        {"saqa_config": {"type": "string_match"}}
    ) == BenchmarkConfig(saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH))
    assert BenchmarkConfig.from_dict(
        {"laqa_config": {"type": "free_form"}}
    ) == BenchmarkConfig(laqa_config=LAQAConfig(type=LongAnswerType.FREE_FORM))


def test_format_config() -> None:
    assert BenchmarkConfig.from_dict(
        {
            "format": {
                "instructions": {
                    "system_prompt_template": "You are a helpful assistant.",
                    "base_inst_template": "Answer the question.",
                    "chat_inst_template": "Think step by step.",
                },
                "prompt": {
                    "question_template": "Q: {{ question }}",
                    "answer_template": "A: {{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            }
        }
    ) == BenchmarkConfig(
        format=FormatConfig(
            instructions=InstructionsConfig(
                system_prompt_template="You are a helpful assistant.",
                base_inst_template="Answer the question.",
                chat_inst_template="Think step by step.",
            ),
            prompt=PromptConfig(
                question_template="Q: {{ question }}",
                answer_template="A: {{ answer }}",
                prompt_template="{{ instruction }}\n\n{{ question }}",
            ),
        )
    )


def test_source_huggingface() -> None:
    assert BenchmarkConfig.from_dict(
        {
            "source": {
                "huggingface": {
                    "path": "AI4Sec/cti-bench",
                    "subset_name": "cti-rcm",
                    "test_split": "test",
                },
                "options": {
                    "dataframe_transform_expr": "df[['question', 'answer']]",
                },
            }
        }
    ) == BenchmarkConfig(
        source=SourceConfig(
            huggingface=HuggingFaceSourceConfig(
                path="AI4Sec/cti-bench",
                subset_name="cti-rcm",
                test_split="test",
            ),
            options=SourceOptionsConfig(
                dataframe_transform_expr="df[['question', 'answer']]",
            ),
        )
    )


def test_output_processing_with_answer_formats() -> None:
    assert BenchmarkConfig.from_dict(
        {
            "output_processing": {
                "primary_metric": "accuracy.mean",
                "answer_formats": [
                    {
                        "pattern": r"(?i)\b(cwe[-]\d+)\b",
                        "capture_transform": {
                            "params": ["x"],
                            "expr": "x.upper()",
                        },
                        "disambiguation": "match_if_unique",
                        "format_type": "proper",
                    },
                    {
                        "pattern": r"(?i)\b(?:cwe|id):\s+(\d+)\b",
                        "format_type": "improper",
                    },
                ],
            }
        }
    ) == BenchmarkConfig(
        output_processing=OutputProcessingConfig(
            primary_metric="accuracy.mean",
            answer_formats=[
                PatternDef(
                    pattern=r"(?i)\b(cwe[-]\d+)\b",
                    capture_transform=CaptureTransform(params=["x"], expr="x.upper()"),
                    disambiguation=Disambiguation.MATCH_IF_UNIQUE,
                    format_type=AnswerFormat.PROPER,
                ),
                PatternDef(
                    pattern=r"(?i)\b(?:cwe|id):\s+(\d+)\b",
                    format_type=AnswerFormat.IMPROPER,
                ),
            ],
        )
    )


def test_output_processing_with_score_fns() -> None:
    assert BenchmarkConfig.from_dict(
        {
            "output_processing": {
                "score_fns": {
                    "jaccard_index": {"type": "jaccard"},
                    "llm_judge": {
                        "type": "llm_judge",
                        "judge_model": {"model_path": "gpt-4"},
                        "verdict_formats": [{"pattern": ".*", "format_type": "proper"}],
                    },
                },
            }
        }
    ) == BenchmarkConfig(
        output_processing=OutputProcessingConfig(
            score_fns={
                "jaccard_index": ScoreFnConfig(type="jaccard", kwargs={}),
                "llm_judge": ScoreFnConfig(
                    type="llm_judge",
                    kwargs={
                        "judge_model": {"model_path": "gpt-4"},
                        "verdict_formats": [{"pattern": ".*", "format_type": "proper"}],
                    },
                ),
            },
        )
    )


def test_full_config() -> None:
    """Test a realistic full benchmark config similar to ctibench-rcm."""
    assert BenchmarkConfig.from_dict(
        {
            "metadata": {
                "name": "ctibench-rcm",
                "description": "Root Cause Mapping benchmark.",
                "urls": ["https://arxiv.org/abs/2406.07599"],
                "categories": ["security"],
                "state": "enabled",
            },
            "source": {
                "huggingface": {
                    "path": "AI4Sec/cti-bench",
                    "subset_name": "cti-rcm",
                    "test_split": "test",
                },
                "options": {
                    "dataframe_transform_expr": "df[['question', 'answer']]",
                },
            },
            "saqa_config": {"type": "string_match"},
            "format": {
                "instructions": {
                    "base_inst_template": "Map the CVE to a CWE ID.",
                    "chat_inst_template": "Analyze and map to CWE.",
                },
                "prompt": {
                    "question_template": "{{ question }}",
                    "answer_template": "{{ answer }}",
                    "prompt_template": "{{ instruction }}\n\n{{ question }}",
                },
            },
            "output_processing": {
                "primary_metric": "accuracy.mean",
                "answer_formats": [
                    {
                        "pattern": r"(?i)\b(cwe[-]\d+)\b",
                        "capture_transform": {"params": ["x"], "expr": "x.upper()"},
                        "disambiguation": "match_if_unique",
                        "format_type": "proper",
                    },
                ],
            },
        }
    ) == BenchmarkConfig(
        metadata=MetadataConfig(
            name="ctibench-rcm",
            description="Root Cause Mapping benchmark.",
            urls=["https://arxiv.org/abs/2406.07599"],
            categories=["security"],
            state=BenchmarkState.ENABLED,
        ),
        source=SourceConfig(
            huggingface=HuggingFaceSourceConfig(
                path="AI4Sec/cti-bench",
                subset_name="cti-rcm",
                test_split="test",
            ),
            options=SourceOptionsConfig(
                dataframe_transform_expr="df[['question', 'answer']]",
            ),
        ),
        saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH),
        format=FormatConfig(
            instructions=InstructionsConfig(
                base_inst_template="Map the CVE to a CWE ID.",
                chat_inst_template="Analyze and map to CWE.",
            ),
            prompt=PromptConfig(
                question_template="{{ question }}",
                answer_template="{{ answer }}",
                prompt_template="{{ instruction }}\n\n{{ question }}",
            ),
        ),
        output_processing=OutputProcessingConfig(
            primary_metric="accuracy.mean",
            answer_formats=[
                PatternDef(
                    pattern=r"(?i)\b(cwe[-]\d+)\b",
                    capture_transform=CaptureTransform(params=["x"], expr="x.upper()"),
                    disambiguation=Disambiguation.MATCH_IF_UNIQUE,
                    format_type=AnswerFormat.PROPER,
                ),
            ],
        ),
    )


def test_mutually_exclusive_config_types() -> None:
    with pytest.raises(
        ValueError,
        match="At most one of mcqa_config, saqa_config, laqa_config may be provided.",
    ):
        BenchmarkConfig(
            mcqa_config=MCQAConfig(),
            saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH),
        )
    with pytest.raises(
        ValueError,
        match="At most one of mcqa_config, saqa_config, laqa_config may be provided.",
    ):
        BenchmarkConfig(
            mcqa_config=MCQAConfig(),
            laqa_config=LAQAConfig(type=LongAnswerType.FREE_FORM),
        )
    with pytest.raises(
        ValueError,
        match="At most one of mcqa_config, saqa_config, laqa_config may be provided.",
    ):
        BenchmarkConfig(
            mcqa_config=MCQAConfig(),
            saqa_config=SAQAConfig(type=ShortAnswerType.STRING_MATCH),
            laqa_config=LAQAConfig(type=LongAnswerType.FREE_FORM),
        )
