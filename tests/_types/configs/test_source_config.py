# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SourceConfig validation."""

import pytest

from faith._types.configs.source import (
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
            files=FilesSourceConfig(),
        )
    with pytest.raises(
        ValueError,
        match="At most one of huggingface, files, git_repo may be provided.",
    ):
        SourceConfig(
            huggingface=HuggingFaceSourceConfig(path="some/path"),
            git_repo=GitRepoSourceConfig(),
        )
    with pytest.raises(
        ValueError,
        match="At most one of huggingface, files, git_repo may be provided.",
    ):
        SourceConfig(
            files=FilesSourceConfig(),
            git_repo=GitRepoSourceConfig(),
        )
    with pytest.raises(
        ValueError,
        match="At most one of huggingface, files, git_repo may be provided.",
    ):
        SourceConfig(
            huggingface=HuggingFaceSourceConfig(path="some/path"),
            files=FilesSourceConfig(),
            git_repo=GitRepoSourceConfig(),
        )
