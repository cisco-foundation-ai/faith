# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""In-memory tar construction shared by concrete environments."""

import io
import shlex
import tarfile
from pathlib import Path


def workspace_operand(name: str) -> str:
    """Quote a workspace-root file name for use as a shell command operand.

    Prefixed with ``./`` so a name beginning with ``-`` is never parsed as an
    option; unlike ``--``, this works for ``unzip``, which has no end-of-options
    marker.
    """
    return shlex.quote(f"./{name}")


def tar_bytes(members: list[tuple[str, bytes]]) -> bytes:
    """Tar (workspace-relative path, content) pairs in memory."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for path, content in members:
            info = tarfile.TarInfo(name=path)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def tar_directory(data_path: Path) -> bytes:
    """Tar a local directory's contents in memory, rooted at the directory itself.

    Entries are added non-recursively in sorted order so every directory, symlink,
    and file mode is preserved verbatim: an executable stays executable, and a
    symlink is archived as a link rather than having its target's content inlined
    under the link's name.
    """
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for path in sorted(data_path.rglob("*")):
            tar.add(
                path, arcname=path.relative_to(data_path).as_posix(), recursive=False
            )
    return stream.getvalue()
