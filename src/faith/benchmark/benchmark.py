# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""The foundation for a benchmark of question-answer data and evaluation criteria."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase

from faith import __version__
from faith._internal.algo.sampling import NShotSampler
from faith._types.benchmark.spec import BenchmarkSpec
from faith._types.config.benchmark import BenchmarkConfig
from faith._types.config.patterns import PatternDef
from faith._types.model.generation import GenerationMode
from faith.benchmark.dataset.dataset import BenchmarkDataset
from faith.benchmark.dataset.load import load_data, sample_datasets
from faith.benchmark.formatting.qa import QAFormatter
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader

logger = logging.getLogger(__name__)


class Benchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(
        self,
        spec: BenchmarkSpec,
        config: BenchmarkConfig,
        path: Path | None = None,
        seed: int | None = None,
    ):
        """Initialize the benchmark's shared identity.

        Args:
            spec (BenchmarkSpec): The specification for the benchmark.
            config (BenchmarkConfig): The configuration for the benchmark.
            path (Path | None): The path to the benchmark data directory used when
                loading data.
            seed (int | None): The random seed for reproducibility.
        """
        self._name = spec.name
        self._config = config
        self._seed = seed
        self._path = path

    @property
    def name(self) -> str:
        """Fetch the name of the benchmark."""
        return self._name

    @property
    def version(self) -> str:
        """Fetch the version of the benchmark."""
        return __version__

    @abstractmethod
    def log_grader(
        self,
        *,
        model_format_config: PatternDef | None = None,
    ) -> LogGrader:
        """Fetch a log grader that scores this benchmark's logged records."""

    @abstractmethod
    def grade_aggregator(self) -> GradeAggregator:
        """Fetch a grade aggregator that summarizes this benchmark's graded records."""


class QABenchmark(Benchmark):
    """Base class for all question-answer benchmarks."""

    def __init__(
        self,
        spec: BenchmarkSpec,
        config: BenchmarkConfig,
        path: Path | None = None,
        seed: int | None = None,
    ):
        """Initialize the QA benchmark.

        Args:
            spec (BenchmarkSpec): The specification for the benchmark.
            config (BenchmarkConfig): The configuration for the benchmark.
            path (Path | None): The path to the benchmark data directory used when
                loading data.
            seed (int | None): The random seed for reproducibility.
        """
        super().__init__(spec, config, path=path, seed=seed)
        self._generation_mode = spec.generation_mode
        self._n_shot = spec.n_shot
        self._formatter = QAFormatter(spec.prompt_format, self._config.format)

    @property
    def answer_set(self) -> frozenset[str] | None:
        """Fetch the space of answer symbols for the benchmark or None if not applicable."""
        return None

    @property
    def generation_mode(self) -> GenerationMode:
        """Fetch the generation mode used for this benchmark."""
        return self._generation_mode

    @property
    def formatter(self) -> QAFormatter:
        """Fetch the benchmark's formatter used for creating prompts."""
        return self._formatter

    def answer_leadin(self, tokenizer: PreTrainedTokenizerBase) -> str:
        """Fetch the answer lead-in for the benchmark.

        This is required for logits and next-token prediction tasks to seed the response
        generation with the correct answer format.
        """
        raise NotImplementedError(
            f"Benchmark {self.name} does not implement answer lead-in."
        )

    def build_dataset(
        self,
        sample_size: int | None = None,
        randomize_choices: bool = False,
    ) -> BenchmarkDataset:
        """Builds the dataset for this benchmark.

        Loads the benchmark's data from its source, samples it, splits off the
        n-shot holdout, then delegates the type-specific construction to
        `_build_dataset`.
        """
        # Build the benchmark's dataset from the benchmark configuration.
        if self._seed is None:
            logger.warning("No seed provided for benchmark; using default seed.")
        rng = np.random.default_rng(self._seed)
        ancillary_columns = frozenset(self._config.source.ancillary_columns)
        benchdata, holdout = load_data(self.name, self._path, self._config.source)
        benchdata, holdout = sample_datasets(
            benchdata, holdout, self._n_shot, sample_size, rng
        )
        bench_cols = frozenset(benchdata.columns)
        assert (
            ancillary_columns <= bench_cols
        ), f"Specified ancillary columns {list(ancillary_columns - bench_cols)} are not present in the benchmark data."
        if holdout is not None:
            assert bench_cols == frozenset(
                holdout.columns
            ), "Benchmark data and holdout data must have the same columns."
        nshot_sampler = NShotSampler(holdout, self._n_shot, rng)
        return self._build_dataset(
            benchdata, nshot_sampler, rng, randomize_choices, ancillary_columns
        )

    @abstractmethod
    def _build_dataset(
        self,
        benchmark_data: pd.DataFrame,
        nshot_sampler: NShotSampler,
        rng: np.random.Generator,
        randomize_choices: bool = False,
        ancillary_columns: frozenset[str] = frozenset(),
    ) -> BenchmarkDataset:
        """Builds the dataset for this benchmark, returning a BenchmarkDataset object."""
