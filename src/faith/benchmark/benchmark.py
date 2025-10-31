# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""The foundation for a benchmark of question-answer data and evaluation criteria."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase

from faith import __version__
from faith._internal.algo.sampling import NShotSampler
from faith._internal.types.flags import GenerationMode
from faith.benchmark.dataset.dataset import BenchmarkDataset
from faith.benchmark.dataset.load import load_data, sample_datasets
from faith.benchmark.formatting.qa import QAFormatter
from faith.benchmark.grading.grade_aggregator import GradeAggregator
from faith.benchmark.grading.log_grader import LogGrader
from faith.benchmark.types import BenchmarkSpec

logger = logging.getLogger(__name__)


class Benchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(
        self,
        spec: BenchmarkSpec,
        config: dict[str, Any],
        path: Path | None = None,
        seed: int | None = None,
    ):
        """Initialize the Benchmark class.

        Args:
            spec (BenchmarkSpec): The specification for the benchmark.
            config (dict[str, Any]): The configuration dictionary for the benchmark.
            formatter (QAFormatter): The formatter used to create prompts.
            path (Path | None): The path to the benchmark data directory used when
                loading data.
            seed (int | None): The random seed for reproducibility.
        """
        self._name = spec.name
        self._generation_mode = spec.generation_mode
        self._n_shot = spec.n_shot
        self._config = config
        self._formatter = QAFormatter(spec.prompt_format, self._config["format"])
        self._seed = seed
        self._path = path

    @property
    def answer_set(self) -> frozenset[str] | None:
        """Fetch the space of answer symbols for the benchmark or None if not applicable."""
        return None

    @property
    def generation_mode(self) -> GenerationMode:
        """Fetch the generation mode used for this benchmark."""
        return self._generation_mode

    @property
    def name(self) -> str:
        """Fetch the name of the benchmark."""
        return self._name

    @property
    def formatter(self) -> QAFormatter:
        """Fetch the benchmark's formatter used for creating prompts."""
        return self._formatter

    @property
    def version(self) -> str:
        """Fetch the version of the benchmark."""
        return __version__

    def answer_leadin(self, tokenizer: PreTrainedTokenizerBase) -> str:
        """Fetch the answer lead-in for the benchmark.

        This is required for logits and next-token prediction tasks to seed the response
        generation with the correct answer format.
        """
        raise NotImplementedError(
            f"Benchmark {self.name} does not implement answer lead-in."
        )

    @abstractmethod
    def build_dataset(
        self,
        sample_size: int | None = None,
        randomize_choices: bool = False,
    ) -> "BenchmarkDataset":
        """Builds the dataset for this benchmark."""

    @abstractmethod
    def log_grader(
        self, model_format_config: dict[str, Any], recompute_stats: bool = False
    ) -> LogGrader:
        """Fetch a log grader for this benchmark."""

    @abstractmethod
    def grade_aggregator(self) -> GradeAggregator:
        """Fetch a grade aggregator for this benchmark."""


class BaseBenchmark(Benchmark):
    """Base class for benchmarks that uses a BenchmarkDataset."""

    def build_dataset(
        self,
        sample_size: int | None = None,
        randomize_choices: bool = False,
    ) -> BenchmarkDataset:
        """Builds the dataset for this benchmark."""
        # Build the benchmark's dataset from the benchmark configuration.
        if self._seed is None:
            logger.warning("No seed provided for benchmark; using default seed.")
        rng = np.random.default_rng(self._seed)
        benchdata, holdout = load_data(self.name, self._path, self._config["source"])
        benchdata, holdout = sample_datasets(
            benchdata, holdout, self._n_shot, sample_size, rng
        )
        if holdout is not None:
            assert set(benchdata.columns) == set(
                holdout.columns
            ), "Benchmark data and holdout data must have the same columns."
        nshot_sampler = NShotSampler(holdout, self._n_shot, rng)
        return self._build_dataset(benchdata, nshot_sampler, rng, randomize_choices)

    @abstractmethod
    def _build_dataset(
        self,
        benchmark_data: pd.DataFrame,
        nshot_sampler: NShotSampler,
        rng: np.random.Generator,
        randomize_choices: bool = False,
    ) -> BenchmarkDataset:
        """Builds the dataset for this benchmark, returning a BenchmarkDataset object."""
