# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorkspaceLayout, which derives the agent's working directory."""

import pytest

from faith.agentic.environments._workspace import WorkspaceLayout


@pytest.mark.parametrize(
    ("workspace_dir", "working_subdir", "working_dir"),
    [
        ("/workspace", "", "/workspace"),
        ("/workspace", "repo", "/workspace/repo"),
        ("/workspace", "repo/", "/workspace/repo"),
        ("/workspace", "a/b", "/workspace/a/b"),
        ("/app/", "repo", "/app/repo"),
    ],
    ids=["no_subdir", "subdir", "trailing_slash", "nested", "custom_root"],
)
def test_the_working_dir_joins_the_root_and_the_subdir(
    workspace_dir: str, working_subdir: str, working_dir: str
) -> None:
    assert WorkspaceLayout(workspace_dir, working_subdir).working_dir == working_dir
