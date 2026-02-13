# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from faith._internal.io.datastore import (
    DataStoreContext,
    GCPDataStore,
    LocalDataStore,
    ReadOnlyDataContext,
    _cp_cmd_args,
    _ensure_trailing_slash,
    _rsync_cmd_args,
    resolve_storage_path,
)

_EXPECTED_BAR_TXT_CONTENT = """# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

Kermode
"""


def assert_directory_contents(path: Path, expected_files: set[str]) -> None:
    """Assert the given path contains exactly the expected files (relative to the path)."""
    assert path.exists() and path.is_dir(), f"Expected {path} to be a directory"
    assert set(path.rglob("*")) == {path / f for f in expected_files}


def test_rsync_cmd_args() -> None:
    assert _rsync_cmd_args("gs://test/fake-ds", "/tmp/foo") == [
        "gsutil",
        "-m",
        "rsync",
        "-R",
        "-J",
        "-P",
        "gs://test/fake-ds/",
        "/tmp/foo/",
    ]


def test_cp_cmd_args() -> None:
    assert _cp_cmd_args("gs://test/fake-ds/bar.txt", "/tmp/foo") == [
        "gsutil",
        "cp",
        "-J",
        "gs://test/fake-ds/bar.txt",
        "/tmp/foo",
    ]


def test_local_data_store() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        datastore = LocalDataStore(Path(temp_dir) / "datastore")
        assert datastore.pull().name == "datastore"
        datastore.push()

        sub_store = datastore.sub_store(Path("subdir"))
        assert sub_store.pull().name == "subdir"
        sub_store.push()


def test_gcp_data_store_directory_pull_and_push() -> None:
    # Test the GCPDataStore class for pulling and pushing data by modifying the rsync
    # command arguments to read and write to local files instead of GCP storage.
    fake_gcp_addr = "gs://test/fake-ds"
    fake_ds_path = str(Path(__file__).parent / "testdata" / "fake_gcp_datastore")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with (
            patch(
                "faith._internal.io.datastore._rsync_cmd_args",
                side_effect=lambda src, dest: [
                    "rsync",
                    "-a",
                    _ensure_trailing_slash(src.replace(fake_gcp_addr, fake_ds_path)),
                    _ensure_trailing_slash(dest.replace(fake_gcp_addr, str(temp_path))),
                ],
            ),
            patch(
                "faith._internal.io.datastore.GCPDataStore._is_file", return_value=False
            ),
        ):
            datastore = GCPDataStore(fake_gcp_addr)

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


def test_gcp_data_store_file_pull_and_push() -> None:
    # Test the GCPDataStore class for pulling and pushing a file by modifying the cp
    # command arguments to read and write to local files instead of GCP storage.
    fake_gcp_addr = "gs://test/fake-ds"
    fake_ds_path = str(Path(__file__).parent / "testdata" / "fake_gcp_datastore")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with (
            patch(
                "faith._internal.io.datastore.GCPDataStore._is_file", return_value=True
            ),
            patch(
                "faith._internal.io.datastore._cp_cmd_args",
                side_effect=lambda src, dest: [
                    "cp",
                    src.replace(fake_gcp_addr, fake_ds_path),
                    dest.replace(fake_gcp_addr, str(temp_path)),
                ],
            ),
        ):
            datastore = GCPDataStore("gs://test/fake-ds/bar.txt")
            assert datastore.pull().read_text() == _EXPECTED_BAR_TXT_CONTENT
            datastore.push()
            assert (temp_path / "bar.txt").read_text() == _EXPECTED_BAR_TXT_CONTENT


def test_gcp_datastore_directory_pull_failure() -> None:
    # Test the GCPDataStore class for pulling data with a command that will always fail.
    with (
        # Mock the rsync command arguments to always fail.
        patch("faith._internal.io.datastore._rsync_cmd_args", return_value=["false"]),
        patch("faith._internal.io.datastore.GCPDataStore._is_file", return_value=False),
    ):
        datastore = GCPDataStore("gs://test/fake-ds")
        with pytest.raises(
            ValueError,
            match="Failed to rsync from 'gs://test/fake-ds'",
        ):
            datastore.pull()


def test_gcp_datastore_file_pull_failure() -> None:
    # Test the GCPDataStore class for pulling data with a command that will always fail.
    with (
        # Mock the rsync command arguments to always fail.
        patch("faith._internal.io.datastore._cp_cmd_args", return_value=["false"]),
        patch("faith._internal.io.datastore.GCPDataStore._is_file", return_value=True),
    ):
        datastore = GCPDataStore("gs://test/fake-ds/bar.txt")
        with pytest.raises(
            ValueError,
            match="Failed to copy from 'gs://test/fake-ds/bar.txt'",
        ):
            datastore.pull()


def test_gcp_datastore_is_file_true() -> None:
    # Test _is_file returns True when gsutil ls -d -l reports an object.
    # pylint: disable=protected-access
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

        assert GCPDataStore._is_file("gs://test-bucket/fake-datastore/bar.txt") is True
        mock_popen.assert_called_once_with(
            ["gsutil", "ls", "-d", "-l", "gs://test-bucket/fake-datastore/bar.txt"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )


def test_gcp_datastore_is_file_false() -> None:
    # Test _is_file returns False when gsutil ls -d -l reports a prefix (directory).
    # pylint: disable=protected-access
    dir_output = "                                 gs://test-bucket/fake-datastore/"
    with patch("faith._internal.io.datastore.Popen") as mock_popen:
        mock_process = Mock(
            communicate=Mock(return_value=(dir_output, "")),
            returncode=0,
        )
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        assert GCPDataStore._is_file("gs://test-bucket/fake-datastore") is False


def test_gcp_datastore_is_file_failure() -> None:
    # Test _is_file raises ValueError when gsutil ls -d -l fails.
    # pylint: disable=protected-access
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
            GCPDataStore._is_file("gs://test-bucket/bad-path")


def test_datastore_context() -> None:
    # Test the DataStoreContext class with a local path.
    with DataStoreContext(
        str(Path(__file__).parent / "testdata" / "fake_gcp_datastore")
    ) as ds:
        assert isinstance(ds, LocalDataStore)
        assert_directory_contents(ds.pull(), {"bar.txt", "sub", "sub/foo.txt"})

    # Test the DataStoreContext class with a fake GCP path.
    fake_gcp_addr = "gs://test-bucket/fake-datastore"
    fake_ds_path = str(Path(__file__).parent / "testdata" / "fake_gcp_datastore")
    with tempfile.TemporaryDirectory() as temp_dir:
        with (
            patch(
                "faith._internal.io.datastore._rsync_cmd_args",
                side_effect=lambda src, dest: [
                    "rsync",
                    "-a",
                    _ensure_trailing_slash(src.replace(fake_gcp_addr, fake_ds_path)),
                    _ensure_trailing_slash(dest.replace(fake_gcp_addr, str(temp_dir))),
                ],
            ),
            patch(
                "faith._internal.io.datastore.GCPDataStore._is_file", return_value=False
            ),
            DataStoreContext("gs://test-bucket/fake-datastore") as ds,
        ):
            assert isinstance(ds, GCPDataStore)
            assert_directory_contents(ds.pull(), {"bar.txt", "sub", "sub/foo.txt"})


def test_read_only_data_context() -> None:
    # Test the ReadOnlyDataContext class with a local path.
    with ReadOnlyDataContext(
        str(Path(__file__).parent / "testdata" / "fake_gcp_datastore" / "bar.txt"),
    ) as data_path:
        assert data_path.exists() and data_path.is_file()
        assert data_path.name == "bar.txt"

    with (
        patch("faith._internal.io.datastore._rsync_cmd_args") as mock_rsync_cmd_args,
        patch("faith._internal.io.datastore.GCPDataStore._is_file", return_value=False),
    ):
        # Mock the rsync command arguments to read from local files instead of GCP.
        mock_rsync_cmd_args.side_effect = lambda src, dest: [
            "rsync",
            "-a",
            str(Path(__file__).parent / "testdata" / "fake_gcp_datastore") + "/",
            str(dest),
        ]
        with ReadOnlyDataContext("gs://test-bucket/fake-datastore") as data_path:
            assert_directory_contents(data_path, {"bar.txt", "sub", "sub/foo.txt"})


# Note: The following test does not test the use of rsync to GCP storage, only the
# scaffolding of the GCP synchronizer and the resolve_storage_path function.
def test_resolve_storage_path() -> None:
    # Test with a command that will always fail.
    with (
        patch("faith._internal.io.datastore._rsync_cmd_args", return_value=["false"]),
        pytest.raises(ValueError, match="Initial upload to GCP failed"),
    ):
        resolve_storage_path("gs://test-bucket")

    # Test with a command that will always succeed.
    with (
        patch("faith._internal.io.datastore._rsync_cmd_args", return_value=["true"]),
        resolve_storage_path("gs://test-bucket") as path,
    ):
        assert path.exists() and path.is_dir()

    with resolve_storage_path("/tmp/test-dir") as path:
        assert str(path) == "/tmp/test-dir"
