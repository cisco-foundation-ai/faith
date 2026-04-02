# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pytest_unordered import unordered

from faith._internal.io.resource_provider import ResourceProvider

_EXPECTED_BAR_TXT_CONTENT = """# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

Kermode
"""


def _assert_directory_contents(path: Path, expected_files: set[str]) -> None:
    """Assert the given path contains exactly the expected files (relative to the path)."""
    assert path.exists() and path.is_dir(), f"Expected {path} to be a directory"
    assert list(path.rglob("*")) == unordered(path / f for f in expected_files)


def _populate_fake_dir(dest: str) -> None:
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    (d / "a.txt").write_text("a")


def test_local_file() -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        f.write(b"hello")
        f.flush()
        with ResourceProvider(f.name) as path:
            assert path == Path(f.name)
            assert path.read_text(encoding="utf-8") == "hello"


def test_local_directory() -> None:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.txt").write_text("content")
        with ResourceProvider(d) as path:
            _assert_directory_contents(path, expected_files={"a.txt"})


def test_local_missing_raises() -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        with ResourceProvider("/nonexistent/path") as _:
            pass


def test_gcs_file() -> None:
    with (
        patch("faith._internal.io.resource_provider.gcp_is_file", return_value=True),
        patch(
            "faith._internal.io.resource_provider.gcp_cp",
            side_effect=lambda src, dest, **kw: Path(dest)
            .joinpath("bar.txt")
            .write_text(_EXPECTED_BAR_TXT_CONTENT, encoding="utf-8"),
        ),
    ):
        with ResourceProvider("gs://test/fake-ds/bar.txt") as path:
            assert path.name == "bar.txt"
            assert path.read_text(encoding="utf-8") == _EXPECTED_BAR_TXT_CONTENT
    # Temp directory cleaned up after exit.
    assert not path.parent.exists()


def test_gcs_directory() -> None:
    with (
        patch("faith._internal.io.resource_provider.gcp_is_file", return_value=False),
        patch(
            "faith._internal.io.resource_provider.gcp_rsync",
            side_effect=lambda src, dest, **kw: _populate_fake_dir(dest),
        ),
    ):
        with ResourceProvider("gs://test/fake-ds") as path:
            _assert_directory_contents(path, expected_files={"a.txt"})
    # Temp directory cleaned up after exit.
    assert not path.exists()
