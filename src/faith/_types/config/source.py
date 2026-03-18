# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark data source configuration types."""

from dataclasses import dataclass, field
from enum import auto

from dataclasses_json import DataClassJsonMixin, config

from faith._types.enums import CIStrEnum


class DataFileType(CIStrEnum):
    """Enum for the type of data files."""

    CSV = auto()
    JSON = auto()
    JSONL = auto()


@dataclass(frozen=True)
class HuggingFaceSourceConfig(DataClassJsonMixin):
    """Configuration for loading data from HuggingFace datasets."""

    path: str | None = None
    subset_name: str | None = None
    test_split: str | None = None
    dev_split: str | None = None


@dataclass(frozen=True)
class FilesSourceConfig(DataClassJsonMixin):
    """Configuration for loading data from local files."""

    type: DataFileType = field(metadata=config(encoder=str, decoder=DataFileType))
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
class SourceOptionsConfig(DataClassJsonMixin):
    """Options for source data transformation."""

    dataframe_transform_expr: str | None = None


@dataclass(frozen=True)
class SourceConfig(DataClassJsonMixin):
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
