# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-memory tar helpers both concrete environments upload through.

Both DockerEnvironment and KubernetesEnvironment build their uploads here, so the
archive's shape is pinned once in this file; their own tests assert only that the
right archive reaches the right backend call.
"""

import io
import tarfile
from pathlib import Path
from unittest.mock import ANY

import pytest

from faith.agentic.environments._archive import (
    tar_bytes,
    tar_directory,
    workspace_operand,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("repo.zip", "./repo.zip"),
        # A leading '-' would otherwise be parsed as an option by unzip, which
        # has no '--' end-of-options marker.
        ("-rf.zip", "./-rf.zip"),
        ("a b.zip", "'./a b.zip'"),
        ("x;rm -rf /.zip", "'./x;rm -rf /.zip'"),
    ],
)
def test_workspace_operand_quotes_a_name_for_the_shell(
    name: str, expected: str
) -> None:
    assert workspace_operand(name) == expected


def _members(archive: bytes) -> list[tuple[str, int, bytes | None]]:
    """Every member as (name, mode & 0o777, content) — None for a non-file."""
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        return [
            (
                member.name,
                member.mode & 0o777,
                extracted.read() if (extracted := tar.extractfile(member)) else None,
            )
            for member in tar.getmembers()
        ]


def test_tar_bytes_archives_each_pair_at_its_workspace_relative_path() -> None:
    assert _members(tar_bytes([("sub/out.txt", b"data"), ("top", b"")])) == [
        ("sub/out.txt", 0o644, b"data"),
        ("top", 0o644, b""),
    ]


def test_tar_directory_preserves_layout_modes_symlinks_and_empty_dirs(
    tmp_path: Path,
) -> None:
    """The metadata a per-file walk with a default TarInfo would silently drop.

    An executable must stay executable, an empty directory the build system
    expects must survive, and a symlink must be archived as a link rather than
    having its target's content inlined under the link's name. Entries are in
    sorted order, so the same tree always yields the same archive.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.c").write_bytes(b"int main(){}")
    script = tmp_path / "configure"
    script.write_bytes(b"#!/bin/sh\n")
    script.chmod(0o755)
    (tmp_path / "link").symlink_to("configure")
    (tmp_path / "build").mkdir()

    with tarfile.open(fileobj=io.BytesIO(tar_directory(tmp_path))) as tar:
        entries = [
            (m.name, m.mode & 0o777, m.type, m.linkname) for m in tar.getmembers()
        ]
    assert entries == [
        ("build", 0o755, tarfile.DIRTYPE, ""),
        ("configure", 0o755, tarfile.REGTYPE, ""),
        # A symlink's own mode is the OS's business (0o755 on macOS, 0o777 on
        # Linux) and is ignored on extraction; the type and target are the point.
        ("link", ANY, tarfile.SYMTYPE, "configure"),
        ("src", 0o755, tarfile.DIRTYPE, ""),
        ("src/a.c", 0o644, tarfile.REGTYPE, ""),
    ]


def test_tar_directory_archives_contents_not_the_directory_itself(
    tmp_path: Path,
) -> None:
    """Rooted at the directory, so unpacking at the workspace root adds no level."""
    (tmp_path / "a.c").write_bytes(b"int main(){}")
    assert _members(tar_directory(tmp_path)) == [("a.c", 0o644, b"int main(){}")]
