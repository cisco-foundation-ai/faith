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

    path: str
    subset_name: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    test_split: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    dev_split: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )


@dataclass(frozen=True)
class FilesSourceConfig(DataClassJsonMixin):
    """Configuration for loading data from local files."""

    type: DataFileType = field(metadata=config(encoder=str, decoder=DataFileType))
    benchmark_data_paths: list[str] | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    path_glob: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    holdout_data_paths: list[str] | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    selected_columns: list[str] | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )


@dataclass(frozen=True)
class GitRepoSourceConfig(FilesSourceConfig):
    """Configuration for loading data from a git repository."""

    repo_url: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    branch: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    commit: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )

    def __post_init__(self) -> None:
        if not self.repo_url:
            raise ValueError("repo_url is required for GitRepoSourceConfig.")


@dataclass(frozen=True)
class SourceOptionsConfig(DataClassJsonMixin):
    """Options for source data transformation."""

    dataframe_transform_expr: str | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )


@dataclass(frozen=True)
class SourceConfig(DataClassJsonMixin):
    """Configuration for benchmark data sources."""

    huggingface: HuggingFaceSourceConfig | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    files: FilesSourceConfig | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    git_repo: GitRepoSourceConfig | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    options: SourceOptionsConfig | None = field(
        default=None, metadata=config(exclude=lambda x: x is None)
    )
    ancillary_columns: list[str] = field(
        default_factory=list, metadata=config(exclude=lambda x: not x)
    )

    def __post_init__(self) -> None:
        if (
            sum(c is not None for c in [self.huggingface, self.files, self.git_repo])
            > 1
        ):
            raise ValueError(
                "At most one of huggingface, files, git_repo may be provided."
            )
