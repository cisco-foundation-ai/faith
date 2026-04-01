# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pytest_unordered import unordered

from faith._internal.io.datastore import (
    DatastoreContext,
    _ensure_trailing_slash,
    _gcp_cp_args,
    _gcp_is_file,
    _gcp_rsync_args,
)

_EXPECTED_BAR_TXT_CONTENT = """# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

Kermode
"""


def assert_directory_contents(path: Path, expected_files: set[str]) -> None:
    """Assert the given path contains exactly the expected files (relative to the path)."""
    assert path.exists() and path.is_dir(), f"Expected {path} to be a directory"
    assert list(path.rglob("*")) == unordered(path / f for f in expected_files)


def test_gcp_rsync_args() -> None:
    assert _gcp_rsync_args("gs://test/fake-ds", "/tmp/foo") == [
        "gsutil",
        "-m",
        "rsync",
        "-R",
        "-J",
        "-P",
        "gs://test/fake-ds/",
        "/tmp/foo/",
    ]


def test_gcp_cp_args() -> None:
    assert _gcp_cp_args("gs://test/fake-ds/bar.txt", "/tmp/foo") == [
        "gsutil",
        "cp",
        "-J",
        "gs://test/fake-ds/bar.txt",
        "/tmp/foo",
    ]


def test_local_datastore() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with DatastoreContext.from_path(str(Path(temp_dir) / "datastore")) as datastore:
            assert datastore.path == Path(temp_dir) / "datastore"
            assert datastore.pull().name == "datastore"
            datastore.push()

            sub_store = datastore.sub_store(Path("subdir"))
            assert sub_store.pull().name == "subdir"
            sub_store.push()


def test_gcp_datastore_directory_pull_and_push() -> None:
    # Test a GCP Datastore for pulling and pushing data by modifying the rsync
    # command arguments to read and write to local files instead of GCP storage.
    fake_gcp_addr = "gs://test/fake-ds"
    fake_ds_path = str(Path(__file__).parent / "testdata" / "fake_gcp_datastore")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with (
            patch("faith._internal.io.datastore._gcp_is_file", return_value=False),
            patch(
                "faith._internal.io.datastore._gcp_rsync_args",
                side_effect=lambda src, dest: [
                    "rsync",
                    "-a",
                    _ensure_trailing_slash(src.replace(fake_gcp_addr, fake_ds_path)),
                    _ensure_trailing_slash(dest.replace(fake_gcp_addr, str(temp_path))),
                ],
            ),
            DatastoreContext.from_path(fake_gcp_addr) as datastore,
        ):
            # Test the pull method.
            path = datastore.pull()
            assert_directory_contents(path, {"bar.txt", "sub", "sub/foo.txt"})

            # Test the push method.
            datastore.push()
            assert_directory_contents(temp_path, {"bar.txt", "sub", "sub/foo.txt"})

            # Test re-pushing after local modifications.
            (path / "bar.txt").write_text("modified content")
            (path / "new.txt").write_text("new content")
            datastore.push()
            assert_directory_contents(
                temp_path, {"bar.txt", "new.txt", "sub", "sub/foo.txt"}
            )
            assert (temp_path / "bar.txt").read_text() == "modified content"
            assert (temp_path / "new.txt").read_text() == "new content"

            # Construct a sub-store and test its pull and push.
            sub_store = datastore.sub_store(Path("sub"))
            assert_directory_contents(sub_store.pull(), {"foo.txt"})
            sub_store.push()
            assert_directory_contents(temp_path / "sub", {"foo.txt"})


def test_gcp_datastore_file_pull_and_push() -> None:
    # Test a GCP Datastore for pulling and pushing a file by modifying the cp
    # command arguments to read and write to local files instead of GCP storage.
    fake_gcp_addr = "gs://test/fake-ds"
    fake_ds_path = str(Path(__file__).parent / "testdata" / "fake_gcp_datastore")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with (
            patch("faith._internal.io.datastore._gcp_is_file", return_value=True),
            patch(
                "faith._internal.io.datastore._gcp_cp_args",
                side_effect=lambda src, dest: [
                    "cp",
                    src.replace(fake_gcp_addr, fake_ds_path),
                    dest.replace(fake_gcp_addr, str(temp_path)),
                ],
            ),
            DatastoreContext.from_path(
                "gs://test/fake-ds/bar.txt", expect_exists=True
            ) as datastore,
        ):
            assert datastore.pull().read_text() == _EXPECTED_BAR_TXT_CONTENT
            datastore.push()
            assert (temp_path / "bar.txt").read_text() == _EXPECTED_BAR_TXT_CONTENT


def test_gcp_datastore_directory_pull_failure() -> None:
    # Test a GCP Datastore for pulling a directory with a command that will always fail.
    with (
        # Mock the rsync command arguments to always fail.
        patch("faith._internal.io.datastore._gcp_is_file", return_value=False),
        patch("faith._internal.io.datastore._gcp_rsync_args", return_value=["false"]),
        DatastoreContext.from_path("gs://test/fake-ds") as datastore,
    ):
        # Test that pull does not raise when raise_on_error=False (default).
        assert datastore.pull() is not None

        # Test that pull raises a RuntimeError when raise_on_error=True.
        with pytest.raises(RuntimeError, match="Failed to run GCP command"):
            datastore.pull(raise_on_error=True)


def test_gcp_datastore_file_pull_failure() -> None:
    # Test a GCP Datastore for pulling a file with a command that will always fail.
    with (
        # Mock the rsync command arguments to always fail.
        patch("faith._internal.io.datastore._gcp_is_file", return_value=True),
        patch("faith._internal.io.datastore._gcp_cp_args", return_value=["false"]),
        DatastoreContext.from_path(
            "gs://test/fake-ds/bar.txt", expect_exists=True
        ) as datastore,
    ):
        # Test that pull does not raise when raise_on_error=False (default).
        assert datastore.pull() is not None

        # Test that pull raises a RuntimeError when raise_on_error=True.
        with pytest.raises(RuntimeError, match="Failed to run GCP command"):
            datastore.pull(raise_on_error=True)


def test_gcp_is_file_true() -> None:
    # Test _gcp_is_file returns True when gsutil ls -d -l reports an object.
    file_output = """
        1234  2024-01-01T00:00:00Z  gs://test-bucket/fake-datastore/bar.txt
        TOTAL: 1 objects, 1234 bytes (1.21 KiB)
"""
    with patch("faith._internal.io.datastore.Popen") as mock_popen:
        mock_process = Mock(
            communicate=Mock(return_value=(file_output, "")),
            returncode=0,
        )
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        assert _gcp_is_file("gs://test-bucket/fake-datastore/bar.txt") is True
        mock_popen.assert_called_once_with(
            ["gsutil", "ls", "-d", "-l", "gs://test-bucket/fake-datastore/bar.txt"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )


def test_gcp_is_file_false() -> None:
    # Test _gcp_is_file returns False when gsutil ls -d -l reports a prefix (directory).
    dir_output = "                                 gs://test-bucket/fake-datastore/"
    with patch("faith._internal.io.datastore.Popen") as mock_popen:
        mock_process = Mock(
            communicate=Mock(return_value=(dir_output, "")),
            returncode=0,
        )
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        assert _gcp_is_file("gs://test-bucket/fake-datastore") is False


def test_gcp_is_file_failure() -> None:
    # Test _gcp_is_file raises ValueError when gsutil ls -d -l fails.
    with patch("faith._internal.io.datastore.Popen") as mock_popen:
        mock_process = Mock(
            communicate=Mock(return_value=("", "")),
            returncode=1,
        )
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        with pytest.raises(
            ValueError,
            match="Failed to check if GCP path 'gs://test-bucket/bad-path' is a file.",
        ):
            _gcp_is_file("gs://test-bucket/bad-path")
