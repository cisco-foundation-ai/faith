# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from faith._internal.io.json import read_json_file, write_as_json
from faith._internal.types.flags import GenerationMode, SampleRatio
from faith.benchmark.formatting.prompt import PromptFormatter
from faith.benchmark.types import BenchmarkSpec


def test_benchmark_spec_sha256() -> None:
    spec = BenchmarkSpec(
        name="sui-generis",
        generation_mode=GenerationMode.NEXT_TOKEN,
        prompt_format=PromptFormatter.CHAT,
        n_shot=SampleRatio(3, 4),
    )
    assert (
        spec.sha256()
        == "87fd53b7a1e47ec4c9efca86b6322db78ce54dcaf3674f53f074558632b68b86"
    )


def test_benchmark_spec_to_dict() -> None:
    spec = BenchmarkSpec(
        name="stultus",
        generation_mode=GenerationMode.CHAT_COMPLETION,
        prompt_format=PromptFormatter.CHAT,
        n_shot=SampleRatio(1, 2),
    )
    assert spec.to_dict() == {
        "name": "stultus",
        "generation_mode": "chat_comp",
        "prompt_format": "chat",
        "n_shot": "1/2",
    }


@pytest.mark.parametrize(
    "spec",
    [
        BenchmarkSpec(
            name="stultus",
            generation_mode=GenerationMode.LOGITS,
            prompt_format=PromptFormatter.BASE,
            n_shot=SampleRatio(1, 2),
        ),
        BenchmarkSpec(
            name="ecentrici",
            generation_mode=GenerationMode.NEXT_TOKEN,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(5),
        ),
        BenchmarkSpec(
            name="insulsus",
            generation_mode=GenerationMode.CHAT_COMPLETION,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
        ),
    ],
)
def test_benchmark_spec_json_serialization(spec: BenchmarkSpec) -> None:
    """Test the JSON serialization & deserialization of BenchmarkSpec."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "benchmark_spec.json"
        write_as_json(file_path, spec.to_dict())
        loaded_spec = BenchmarkSpec.from_dict(read_json_file(file_path))
        assert spec == loaded_spec
