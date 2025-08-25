# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Base class `BenchmarkDataset` for all benchmark datasets."""
from abc import ABC, abstractmethod
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from faith._internal.algo.sampling import NShotSampler
from faith.benchmark.formatting.qa import QAFormatter, QARecord


class BenchmarkDataset(ABC):
    """Base class for benchmark datasets; ie. the set of questions that compromise it."""

    def __init__(
        self,
        formatter: QAFormatter,
        benchmark_data: pd.DataFrame,
        nshot_sampler: NShotSampler,
        rng: np.random.Generator,
        required_columns: frozenset[str] = frozenset(),
        optional_columns: frozenset[str] = frozenset(),
    ):
        """Initialize the base `BenchmarkDataset` class."""
        all_allowed_columns = required_columns | optional_columns
        assert required_columns <= set(
            benchmark_data.columns
        ), f"Benchmark data must contain the required columns: {required_columns}."
        assert (
            set(benchmark_data.columns) <= all_allowed_columns
        ), f"Benchmark data contains unexpected columns: {set(benchmark_data.columns) - all_allowed_columns}."
        self._formatter = formatter
        self._benchmark_data = benchmark_data
        self._nshot_sampler = nshot_sampler
        self._rng = rng

    def _get_nshot_examples(self) -> Sequence[QARecord]:
        """Get the n-shot examples for the prompt."""
        if (nshot_data := self._nshot_sampler.get_nshot_examples()) is not None:
            return [self._format_qa(i, sample) for i, sample in nshot_data.iterrows()]
        return []

    def iter_data(self) -> Iterator[QARecord]:
        """Iterates over the benchmark dataset, yielding each example as an QARecord object."""
        for index, sample in self._benchmark_data.iterrows():
            yield self._format_qa(index, sample, self._get_nshot_examples())

    @abstractmethod
    def _format_qa(
        self, index: int, sample: pd.Series, examples: Sequence[QARecord] | None = None
    ) -> QARecord:
        """Format an indexed sample (with examples) into a question-answer record."""
