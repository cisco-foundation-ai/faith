# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark data source configuration types."""

from dataclasses import dataclass, field
from enum import Enum


class DataFileType(Enum):
    """Enum for the type of data files."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"

    def __str__(self) -> str:
        """Return the string representation of the enum."""
        return self.value

    @staticmethod
    def from_string(s: str) -> "DataFileType":
        """Convert a string to an DataFileType enum."""
        try:
            return DataFileType[s.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown data file type: {s}") from e


@dataclass(frozen=True)
class HuggingFaceSourceConfig:
    """Configuration for loading data from HuggingFace datasets."""

    path: str | None = None
    subset_name: str | None = None
    test_split: str | None = None
    dev_split: str | None = None


@dataclass(frozen=True)
class FilesSourceConfig:
    """Configuration for loading data from local files."""

    type: DataFileType | None = None
    benchmark_data_paths: list[str] | None = None
    path_glob: str | None = None
    holdout_data_paths: list[str] | None = None
    selected_columns: list[str] | None = None


@dataclass(frozen=True)
class GitRepoSourceConfig(FilesSourceConfig):
    """Configuration for loading data from a git repository."""

    repo_url: str | None = None
    branch: str | None = None
    commit: str | None = None


@dataclass(frozen=True)
class SourceOptionsConfig:
    """Options for source data transformation."""

    dataframe_transform_expr: str | None = None


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for benchmark data sources."""

    huggingface: HuggingFaceSourceConfig | None = None
    files: FilesSourceConfig | None = None
    git_repo: GitRepoSourceConfig | None = None
    options: SourceOptionsConfig | None = None
    ancillary_columns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            sum(c is not None for c in [self.huggingface, self.files, self.git_repo])
            > 1
        ):
            raise ValueError(
                "At most one of huggingface, files, git_repo may be provided."
            )
