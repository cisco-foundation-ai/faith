# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pytest_unordered import unordered

from faith._internal.io.store import StoreContext

_FAKE_GCP_ADDR = "gs://test/fake-ds"
_FAKE_DS_PATH = str(Path(__file__).parent / "testdata" / "fake_gcp_datastore")


def _assert_directory_contents(path: Path, expected_files: set[str]) -> None:
    """Assert the given path contains exactly the expected files (relative to the path)."""
    assert path.exists() and path.is_dir(), f"Expected {path} to be a directory"
    assert list(path.rglob("*")) == unordered(path / f for f in expected_files)


def _fake_rsync(src: str, dest: str, sim_remote: Path, **_kw) -> None:
    """Simulate rsync by copying between local dirs, swapping GCS URLs for real paths."""
    real_src = Path(src.replace(_FAKE_GCP_ADDR, _FAKE_DS_PATH).rstrip("/"))
    real_dest = Path(dest.replace(_FAKE_GCP_ADDR, str(sim_remote)).rstrip("/"))
    if real_src.is_dir():
        real_dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(real_src, real_dest, dirs_exist_ok=True)


def test_local_store() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with StoreContext.from_path(str(Path(temp_dir) / "store")) as store:
            assert store.path == Path(temp_dir) / "store"
            assert store.pull().name == "store"
            store.push()  # no-op

            sub = store.sub_store(Path("subdir"))
            assert sub.path == Path(temp_dir) / "store" / "subdir"
            sub.push()  # no-op


def test_gcp_store_pull_and_push() -> None:
    # Test a GCP Datastore for pulling and pushing data by modifying the rsync
    # command arguments to read and write to local files instead of GCP storage.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with (
            patch(
                "faith._internal.io.store.gcp_rsync",
                side_effect=lambda src, dest, **kw: _fake_rsync(src, dest, temp_path),
            ),
            StoreContext.from_path(_FAKE_GCP_ADDR) as store,
        ):
            # Test the pull method.
            path = store.pull()
            _assert_directory_contents(path, {"bar.txt", "sub", "sub/foo.txt"})

            # Test the push method.
            store.push()
            _assert_directory_contents(temp_path, {"bar.txt", "sub", "sub/foo.txt"})

            # Test re-pushing after local modifications.
            (path / "bar.txt").write_text("modified content")
            (path / "new.txt").write_text("new")
            store.push()
            _assert_directory_contents(
                temp_path, {"bar.txt", "new.txt", "sub", "sub/foo.txt"}
            )
            assert (temp_path / "bar.txt").read_text() == "modified content"
            assert (temp_path / "new.txt").read_text() == "new"

            # Construct a sub-store and test its pull and push.
            sub = store.sub_store(Path("sub"))
            _assert_directory_contents(sub.pull(), {"foo.txt"})
            sub.push()
            _assert_directory_contents(temp_path / "sub", {"foo.txt"})


def test_gcp_store_pull_failure() -> None:
    with (
        patch(
            "faith._internal.io.store.gcp_rsync",
            side_effect=RuntimeError("Failed to run GCP command"),
        ),
        StoreContext.from_path(_FAKE_GCP_ADDR) as store,
    ):
        with pytest.raises(RuntimeError, match="Failed to run GCP command"):
            store.pull(raise_on_error=True)
