# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SourceConfig validation."""

import pytest

from faith._types.config.source import (
    DataFileType,
    FilesSourceConfig,
    GitRepoSourceConfig,
    HuggingFaceSourceConfig,
    SourceConfig,
)


def test_mutually_exclusive_source_types() -> None:
    with pytest.raises(
        ValueError,
        match="At most one of huggingface, files, git_repo may be provided.",
    ):
        SourceConfig(
            huggingface=HuggingFaceSourceConfig(path="some/path"),
            files=FilesSourceConfig(type=DataFileType.CSV),
        )
    with pytest.raises(
        ValueError,
        match="At most one of huggingface, files, git_repo may be provided.",
    ):
        SourceConfig(
            huggingface=HuggingFaceSourceConfig(path="some/path"),
            git_repo=GitRepoSourceConfig(
                type=DataFileType.CSV, repo_url="http://fake.org/repo.git"
            ),
        )
    with pytest.raises(
        ValueError,
        match="At most one of huggingface, files, git_repo may be provided.",
    ):
        SourceConfig(
            files=FilesSourceConfig(type=DataFileType.CSV),
            git_repo=GitRepoSourceConfig(
                type=DataFileType.CSV, repo_url="http://fake.org/repo.git"
            ),
        )
    with pytest.raises(
        ValueError,
        match="At most one of huggingface, files, git_repo may be provided.",
    ):
        SourceConfig(
            huggingface=HuggingFaceSourceConfig(path="some/path"),
            files=FilesSourceConfig(type=DataFileType.CSV),
            git_repo=GitRepoSourceConfig(
                type=DataFileType.CSV, repo_url="http://fake.org/repo.git"
            ),
        )
