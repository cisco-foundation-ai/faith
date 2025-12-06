# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Base class `BenchmarkDataset` for all benchmark datasets."""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Sequence

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
        ancillary_columns: frozenset[str] = frozenset(),
        optional_columns: frozenset[str] = frozenset(),
    ):
        """Initialize the base `BenchmarkDataset` class."""
        benchmark_cols = set(benchmark_data.columns)
        all_req_cols = required_columns | ancillary_columns
        all_allowed_cols = required_columns | ancillary_columns | optional_columns
        assert (
            all_req_cols <= benchmark_cols
        ), f"Benchmark data must contain the required columns: {all_req_cols}."
        assert (
            benchmark_cols <= all_allowed_cols
        ), f"Benchmark data contains unexpected columns: {benchmark_cols - all_allowed_cols}."
        self._formatter = formatter
        self._benchmark_data = benchmark_data
        self._nshot_sampler = nshot_sampler
        self._rng = rng
        self._ancilliary_columns = ancillary_columns

    def _extract_ancillary_data(self, sample: pd.Series) -> dict[str, Any] | None:
        """Get the ancillary columns for the benchmark dataset."""
        if not self._ancilliary_columns:
            return None
        return {col: sample.get(col, None) for col in self._ancilliary_columns}

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
