# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Factory functions to construct benchmark objects from specifications."""

from typing import Any

from faith.benchmark.benchmark import Benchmark
from faith.benchmark.categories.multiple_choice import MCBenchmark
from faith.benchmark.categories.short_answer import SABenchmark
from faith.benchmark.types import BenchmarkSpec


def load_benchmark(
    spec: BenchmarkSpec, config: dict[str, Any], **kwargs: Any
) -> Benchmark:
    """Load a benchmark from a benchmark specification `spec` and configuration `config`."""
    if config.get("mcqa_config", None) is not None:
        return MCBenchmark(spec, config, **kwargs)
    if config.get("saqa_config", None) is not None:
        return SABenchmark(spec, config, **kwargs)
    raise ValueError(f"Unsupported benchmark type for {spec.name}.")
