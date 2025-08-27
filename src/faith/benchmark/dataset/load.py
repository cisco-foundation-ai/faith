# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Load benchmark datasets from various sources and transform them into a standard schema."""
import ast
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

import git
import numpy as np
import pandas as pd
from datasets import load_dataset

from faith._internal.algo.sampling import sample_partition
from faith._internal.parsing.expr import evaluate_expr
from faith._internal.types.flags import SampleRatio

# The maximum comprehension length to allow for data transforms.
MCL_FOR_TRANSFORMS = 10_000_000


class _DataFileType(Enum):
    """Enum for the type of data files."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"

    def __str__(self) -> str:
        """Return the string representation of the enum."""
        return self.value

    @staticmethod
    def from_string(s: str) -> "_DataFileType":
        """Convert a string to an _DataFileType enum."""
        try:
            return _DataFileType[s.upper()]
        except KeyError:
            raise ValueError(f"Unknown data file type: {s}")


def _load_data_files(
    file_glob: Iterable[Path],
    file_type: _DataFileType,
    selected_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load all data files from a glob pattern and return a concatenated DataFrame."""
    if file_type == _DataFileType.CSV:
        dfs = [pd.read_csv(file) for file in file_glob]
    elif file_type == _DataFileType.JSON:
        dfs = [pd.read_json(file) for file in file_glob]
    elif file_type == _DataFileType.JSONL:
        dfs = [pd.read_json(file, lines=True) for file in file_glob]
    assert len(dfs) > 0, f"No {str(file_type)} files found."
    raw_df = pd.concat(dfs, ignore_index=True)

    # If specified, filter the DataFrame to only include selected columns.
    #
    # TODO(https://github.com/RobustIntelligence/faith/issues/195): The inner
    # normalization for JSON files is a workaround for the unusual stucture of
    # Cybermetric's JSON files, which have nested dictionaries;
    # we should consider a more robust solution in the future.
    if selected_columns is not None:
        raw_df = pd.concat(
            [
                pd.json_normalize(raw_df[col])
                if file_type == _DataFileType.JSON
                else raw_df[col]
                for col in selected_columns
            ],
            axis=1,
        )

    return raw_df


def load_data(
    benchmark_name: str,
    benchmark_path: Path | None,
    source_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load the benchmark dataset and transform it into a standard format."""
    dev_df = None
    if "huggingface" in source_cfg:
        hf_config = source_cfg["huggingface"]
        ds = load_dataset(hf_config["path"], hf_config.get("subset_name", None))
        split = ds
        if hf_config.get("test_split", None) is not None:
            split = ds[hf_config["test_split"]]
        df = split.to_pandas()

        # Use the dev split, if specified, for few-shot prompting.
        if hf_config.get("dev_split", None) is not None:
            dev_df = ds[hf_config["dev_split"]].to_pandas()
    elif "files" in source_cfg:
        assert benchmark_path is not None, "Benchmark path must be specified."
        files_cfg = source_cfg["files"]
        df = _load_data_files(
            benchmark_path.glob(files_cfg["path_glob"]),
            _DataFileType.from_string(files_cfg["type"]),
            selected_columns=files_cfg.get("selected_columns", None),
        )
        if "choices" in df.columns:
            df["choices"] = df["choices"].apply(ast.literal_eval)
    elif "git_repo" in source_cfg:
        git_repo_cfg = source_cfg["git_repo"]

        # Clone the git repository to a temporary directory and load selected data.
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            repo = git.Repo.clone_from(
                git_repo_cfg["repo_url"],
                tmp_path,
                branch=git_repo_cfg.get("branch", "main"),
            )
            if git_repo_cfg.get("commit", None) is not None:
                repo.git.checkout(git_repo_cfg["commit"])
            assert (
                len(repo.heads) == 1
            ), f"Expected a single branch, but found {len(repo.heads)} branches."
            df = _load_data_files(
                tmp_path.glob(git_repo_cfg["path_glob"]),
                _DataFileType.from_string(git_repo_cfg["type"]),
                selected_columns=git_repo_cfg.get("selected_columns", None),
            )
    else:
        raise ValueError(
            f"Unsupported source configuration for benchmark {benchmark_name}:\n{source_cfg}"
        )

    # The benchmark-specific transform function is either loaded from the registry
    # or specified in the source configuration options.
    if dt_expr := source_cfg.get("options", {}).get("dataframe_transform_expr", None):
        df = evaluate_expr(
            dt_expr, names={"df": df}, max_comprehension_length=MCL_FOR_TRANSFORMS
        )
        if dev_df is not None:
            dev_df = evaluate_expr(
                dt_expr,
                names={"df": dev_df},
                max_comprehension_length=MCL_FOR_TRANSFORMS,
            )

    assert not df.isna().any().any(), "DataFrame contains NaN values."
    assert (
        not df.isin(["", None]).any().any()
    ), "DataFrame contains empty strings or None values."

    # Convert numpy arrays to lists for JSON serialization; unfortunately, loading
    # datasets from Hugging Face makes lists into numpy arrays.
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
    if dev_df is not None:
        for col in dev_df.columns:
            dev_df[col] = dev_df[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )

    return df, dev_df


def sample_datasets(
    benchdata: pd.DataFrame,
    holdout: pd.DataFrame | None,
    n_shot: SampleRatio,
    sample_size: int | None,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Sample the benchmark data to the specified n-shot size and sample size."""
    # Reshape the data to the specified number of n-shot examples.
    if n_shot.numerator > 0:
        shot_size = n_shot.numerator if n_shot.denominator == 1 else n_shot.denominator
        if holdout is None:
            # If no dev data is provided, sample n-shots from the main data.
            # This is sampled along with the later subsampling based on `sample_size`
            # so that both the selected n-shot examples and subsample are stable.
            total_samples = min(
                (sample_size or len(benchdata)) + shot_size, len(benchdata)
            )
            benchdata, _ = sample_partition(benchdata, total_samples, rng)
            assert (
                len(benchdata) >= shot_size
            ), f"Not enough samples in the benchmark data to create {shot_size}-shot examples."
            holdout, benchdata = benchdata.iloc[:shot_size], benchdata.iloc[shot_size:]

        else:
            # Downsample the dev data to the n-shot size.
            holdout, _ = sample_partition(holdout, shot_size, rng)
    else:
        # If no n-shot examples are needed, set holdout to None.
        holdout = None

    # Samples the dataset to the specified size but retains the original order.
    # Note: sampling with the same seed will yield a subset for all size n < m.
    if sample_size is not None and sample_size < len(benchdata):
        benchdata, _ = sample_partition(benchdata, sample_size, rng)
    benchdata = benchdata.sort_index()

    return benchdata, holdout
