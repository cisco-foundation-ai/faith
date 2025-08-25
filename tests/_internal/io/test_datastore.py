# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from faith._internal.io.datastore import (
    DataStoreContext,
    GCPDataStore,
    GCPSynchronizer,
    LocalDataStore,
    ReadOnlyDataContext,
    resolve_storage_path,
)


def test_local_data_store() -> None:
    # Test the LocalDataStore class.
    with tempfile.TemporaryDirectory() as temp_dir:
        datastore = LocalDataStore(Path(temp_dir) / "datastore")
        path = datastore.pull()
        assert path.name == "datastore"
        datastore.push()

        sub_store = datastore.sub_store(Path("subdir"))
        sub_path = sub_store.pull()
        assert sub_path.name == "subdir"
        sub_store.push()


def test_gcp_data_store_pull_and_push() -> None:
    # Test the GCPDataStore class for pulling and pushing data by modifying the rsync
    # command arguments to read and write to local files instead of GCP storage.
    with patch(
        "faith._internal.io.datastore.GCPDataStore._rsync_cmd_args"
    ) as mock_rsync_cmd_args:
        datastore = GCPDataStore("gs://test-bucket/fake-datastore")

        # Test the pull method.
        mock_rsync_cmd_args.side_effect = lambda src, dest: [
            "rsync",
            "-a",
            str(Path(__file__).parent / "testdata" / "fake_gcp_datastore") + "/",
            str(dest),
        ]
        path = datastore.pull()

        assert path.exists() and path.is_dir()
        assert (
            path / "bar.txt"
        ).exists(), f"Expected `bar.txt` to exist:  {', '.join(str(p) for p in path.rglob('*'))}"
        assert (
            path / "sub" / "foo.txt"
        ).exists(), f"Expected `sub/foo.txt` to exist: {', '.join(str(p) for p in path.rglob('*'))}"

        # Test the push method.
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_rsync_cmd_args.side_effect = lambda src, dest: [
                "rsync",
                "-a",
                str(src) + "/",
                str(temp_dir),
            ]

            datastore.push()

            assert (
                Path(temp_dir) / "bar.txt"
            ).exists(), f"Expected `bar.txt` to exist:  {', '.join(str(p) for p in Path(temp_dir).rglob('*'))}"
            assert (
                Path(temp_dir) / "sub" / "foo.txt"
            ).exists(), f"Expected `sub/foo.txt` to exist: {', '.join(str(p) for p in Path(temp_dir).rglob('*'))}"

        # Construct a sub-store.
        sub_store = datastore.sub_store(Path("sub"))

        # Test the pull method of the sub-store.
        mock_rsync_cmd_args.side_effect = lambda src, dest: [
            "rsync",
            "-a",
            str(Path(__file__).parent / "testdata" / "fake_gcp_datastore" / "sub")
            + "/",
            str(dest),
        ]
        sub_path = sub_store.pull()

        assert sub_path.exists() and sub_path.is_dir()
        assert (
            sub_path / "foo.txt"
        ).exists(), (
            f"Expected `foo.txt` to exist: {', '.join(str(p) for p in path.rglob('*'))}"
        )

        # Test the push method of the sub-store.
        with tempfile.TemporaryDirectory() as temp_sub_dir:
            mock_rsync_cmd_args.side_effect = lambda src, dest: [
                "rsync",
                "-a",
                str(src) + "/",
                str(temp_sub_dir),
            ]

            sub_store.push()

            assert (
                Path(temp_sub_dir) / "foo.txt"
            ).exists(), f"Expected `foo.txt` to exist: {', '.join(str(p) for p in Path(temp_sub_dir).rglob('*'))}"


def test_gcp_datastore_pull_failure() -> None:
    # Test the GCPDataStore class for pulling data with a command that will always fail.
    with patch(
        "faith._internal.io.datastore.GCPDataStore._rsync_cmd_args"
    ) as mock_rsync_cmd_args:
        datastore = GCPDataStore("gs://test-bucket/fake-datastore")

        # Mock the rsync command arguments to always fail.
        mock_rsync_cmd_args.return_value = ["false"]

        with pytest.raises(
            ValueError,
            match="Failed to rsync from 'gs://test-bucket/fake-datastore' to '.*'.",
        ):
            datastore.pull()


def test_gcp_datastore_rsync_cmd_args() -> None:
    # Test the rsync command arguments.
    synchronizer = GCPDataStore("gs://test-bucket/fake-datastore")
    args = synchronizer._rsync_cmd_args("gs://test-bucket/fake-datastore", "/tmp/foo")
    assert args == [
        "gsutil",
        "-m",
        "rsync",
        "-R",
        "-J",
        "-P",
        "gs://test-bucket/fake-datastore",
        "/tmp/foo",
    ]


def test_datastore_context() -> None:
    # Test the DataStoreContext class with a local path.
    with DataStoreContext(
        str(Path(__file__).parent / "testdata" / "fake_gcp_datastore")
    ) as ds:
        assert isinstance(ds, LocalDataStore)
        path = ds.pull()
        assert path.exists() and path.is_dir()
        assert (path / "bar.txt").exists()
        assert (path / "sub" / "foo.txt").exists()

    with patch(
        "faith._internal.io.datastore.GCPDataStore._rsync_cmd_args"
    ) as mock_rsync_cmd_args, DataStoreContext("gs://test-bucket/fake-datastore") as ds:
        # Test the DataStoreContext class with a GCP path.
        assert isinstance(ds, GCPDataStore)

        # Mock the rsync command arguments to read from local files instead of GCP storage.
        mock_rsync_cmd_args.side_effect = lambda src, dest: [
            "rsync",
            "-a",
            str(Path(__file__).parent / "testdata" / "fake_gcp_datastore") + "/",
            str(dest),
        ]

        path = ds.pull()
        assert path.exists() and path.is_dir()
        assert (path / "bar.txt").exists()
        assert (path / "sub" / "foo.txt").exists()


def test_read_only_data_context() -> None:
    # Test the ReadOnlyDataContext class with a local path.
    with ReadOnlyDataContext(
        str(Path(__file__).parent / "testdata" / "fake_gcp_datastore" / "bar.txt"),
        is_file=True,
    ) as data_path:
        assert data_path.name == "bar.txt"
        assert data_path.exists() and data_path.is_file()

    with patch(
        "faith._internal.io.datastore.GCPDataStore._rsync_cmd_args"
    ) as mock_rsync_cmd_args:
        # Mock the rsync command arguments to read from local files instead of GCP.
        mock_rsync_cmd_args.side_effect = lambda src, dest: [
            "rsync",
            "-a",
            str(Path(__file__).parent / "testdata" / "fake_gcp_datastore") + "/",
            str(dest),
        ]
        with ReadOnlyDataContext(
            "gs://test-bucket/fake-datastore", is_file=False
        ) as data_path:
            assert data_path.exists() and data_path.is_dir()
            assert (data_path / "bar.txt").exists()
            assert (data_path / "sub" / "foo.txt").exists()


# Note: The following test does not test the use of rsync to GCP storage, only the
# scaffolding of the GCP synchronizer and the resolve_storage_path function.
@patch("faith._internal.io.datastore.GCPSynchronizer._rsync_cmd_args")
def test_resolve_storage_path(mock_rsync_cmd_args: MagicMock) -> None:
    # Test with a command that will always fail.
    mock_rsync_cmd_args.return_value = ["false"]
    with pytest.raises(ValueError, match="Initial upload to GCP failed"):
        resolve_storage_path("gs://test-bucket")

    # Test with a command that will always succeed.
    mock_rsync_cmd_args.return_value = ["true"]
    with resolve_storage_path("gs://test-bucket") as path:
        assert path.exists() and path.is_dir()

    with resolve_storage_path("/tmp/test-dir") as path:
        assert str(path) == "/tmp/test-dir"


def test_rsync_cmd_args() -> None:
    # Test the rsync command arguments.
    synchronizer = GCPSynchronizer("gs://test-bucket", test_run=False)
    args = synchronizer._rsync_cmd_args()
    assert args == [
        "gsutil",
        "rsync",
        "-R",
        "-J",
        "-P",
        str(synchronizer._temp_dir.name),
        "gs://test-bucket",
    ]
