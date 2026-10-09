# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Where an environment's files live, and where the agent's commands run."""

import posixpath


class WorkspaceLayout:
    """Holds the workspace root and derives the agent's working directory.

    The subdir moves where the agent's commands run, not where files live: the
    backend root, the upload destination, and extraction all stay at
    ``workspace_dir``. A benchmark whose data unpacks into a subdirectory sets it
    so the shell starts where the agent must name files to be scored.
    """

    def __init__(self, workspace_dir: str, working_subdir: str = "") -> None:
        self.workspace_dir = workspace_dir
        self.working_subdir = working_subdir

    @property
    def working_dir(self) -> str:
        """Where the agent's commands run."""
        return posixpath.normpath(
            posixpath.join(self.workspace_dir, self.working_subdir)
        )
