# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""State-machine and safe-upload tests for the bundled sidecar runtime."""

# The reference runtime is a standalone image, not a FAITH Python package.
# pylint: disable=protected-access

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import pytest

_PATH = (
    Path(__file__).parents[3]
    / "src/faith/__benchmarks__/vulnerability-localization/runtime/workspace_policy.py"
)
_SPEC = importlib.util.spec_from_file_location("faith_test_workspace_policy", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_POLICY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_POLICY)


def _policy(directory):
    # Skip the root-only constructor; the live sidecar suite covers it on Linux.
    policy = _POLICY.WorkspacePolicy.__new__(_POLICY.WorkspacePolicy)
    policy.directory = directory
    policy.state_path = directory / "policy.json"
    return policy


def test_writable_policy_remains_writable_until_sealed(tmp_path):
    policy = _policy(tmp_path)
    policy.check_upload()
    policy.check_execution("target:1")
    assert policy.state() is None


def test_policy_requires_restart_persistent_state(tmp_path):
    policy = _policy(tmp_path)
    with patch.object(_POLICY.os.path, "ismount", return_value=False):
        with pytest.raises(_POLICY.PolicyError, match="mounted"):
            policy.seal(123, "target:1")
    assert policy.state() is None


def test_sealing_is_persisted_before_changing_permissions(tmp_path):
    policy = _policy(tmp_path)

    def protect(_fd):
        assert _policy(tmp_path).state() == {
            "status": "sealing",
            "identity": "target:1",
        }

    with (
        patch.object(_POLICY.os.path, "ismount", return_value=True),
        patch.object(_POLICY, "_protect_tree", side_effect=protect) as protect_mock,
    ):
        policy.seal(123, "target:1")
        policy.seal(123, "target:1")
    protect_mock.assert_called_once_with(123)
    restarted = _policy(tmp_path)
    restarted.check_execution("target:1")
    with pytest.raises(_POLICY.PolicyError, match="uploads are disabled"):
        restarted.check_upload()
    with pytest.raises(_POLICY.PolicyError, match="container changed"):
        restarted.check_execution("target:2")


def test_failed_seal_never_reopens_uploads_or_exec(tmp_path):
    policy = _policy(tmp_path)
    with (
        patch.object(_POLICY.os.path, "ismount", return_value=True),
        patch.object(_POLICY, "_protect_tree", side_effect=OSError("interrupted")),
    ):
        with pytest.raises(OSError):
            policy.seal(123, "target:1")
        restarted = _policy(tmp_path)
        with pytest.raises(_POLICY.PolicyError):
            restarted.check_upload()
        with pytest.raises(_POLICY.PolicyError):
            restarted.check_execution("target:1")
        with pytest.raises(_POLICY.PolicyError):
            restarted.seal(123, "target:1")


@pytest.mark.parametrize("content", ["", "{", "{}", "[]", '{"status":"writable"}'])
def test_invalid_persisted_state_fails_closed(tmp_path, content):
    policy = _policy(tmp_path)
    policy.state_path.write_text(content, encoding="utf-8")
    with pytest.raises(_POLICY.PolicyError):
        policy.check_upload()
    with pytest.raises(_POLICY.PolicyError):
        policy.check_execution("target:1")


def test_upload_waiting_for_a_seal_checks_state_after_acquiring_the_lock(tmp_path):
    policy = _policy(tmp_path)
    entered = Event()
    upload = Mock()

    def waiting_upload():
        entered.set()
        with _policy(tmp_path).locked():
            policy.check_upload()
            upload()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with policy.locked():
            future = pool.submit(waiting_upload)
            assert entered.wait(2)
            policy._save({"status": "sealed", "identity": "target:1"})
        with pytest.raises(_POLICY.PolicyError, match="uploads are disabled"):
            future.result(timeout=2)
    upload.assert_not_called()


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a/../../b", "a//b", "./a", ""]
)
def test_upload_rejects_unsafe_names(tmp_path, name):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(_POLICY.PolicyError):
            _POLICY.upload_at(fd, name, BytesIO(b"content"))
    finally:
        os.close(fd)


def test_upload_replaces_links_without_writing_their_targets(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"original")
    (workspace / "symlink").symlink_to(outside)
    os.link(outside, workspace / "hardlink")
    fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in ("symlink", "hardlink"):
            _POLICY.upload_at(fd, name, BytesIO(b"uploaded"))
            assert (workspace / name).read_bytes() == b"uploaded"
            assert outside.read_bytes() == b"original"
        _POLICY.upload_at(fd, "nested/file", BytesIO(b"nested"))
        assert (workspace / "nested/file").read_bytes() == b"nested"
    finally:
        os.close(fd)


def test_upload_refuses_symlink_directories(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "link").symlink_to(tmp_path)
    fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(OSError):
            _POLICY.upload_at(fd, "link/outside", BytesIO(b"content"))
        assert not (tmp_path / "outside").exists()
    finally:
        os.close(fd)


@pytest.mark.parametrize("size", [0, 7, 3 * 1024 * 1024 + 1])
def test_upload_reads_only_bounded_chunks(tmp_path, size):
    class BoundedStream(BytesIO):
        """Fail if an upload attempts to buffer the complete stream."""

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            return super().read(size)

    content = b"x" * size
    source = BoundedStream(content)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _POLICY.upload_at(fd, "file", source)
    finally:
        os.close(fd)
    assert (tmp_path / "file").read_bytes() == content
    assert not source.closed
    assert not list(tmp_path.glob(".faith-upload-*"))


@pytest.mark.parametrize("stage", ["read", "write", "rename"])
def test_failed_upload_preserves_destination_and_removes_temporary(tmp_path, stage):
    destination = tmp_path / "file"
    destination.write_bytes(b"original")
    source: BytesIO | Mock = Mock(
        read=Mock(side_effect=[b"partial", OSError("read failed")])
    )
    real_fdopen = os.fdopen

    @contextmanager
    def failing_writer(fd, mode):
        with real_fdopen(fd, mode) as stream:

            def write(content):
                stream.write(content[:2])
                raise OSError("write failed")

            yield Mock(write=write)

    if stage == "read":
        failure = patch.object(_POLICY.os, "fdopen", wraps=real_fdopen)
    elif stage == "write":
        source = BytesIO(b"content")
        failure = patch.object(_POLICY.os, "fdopen", side_effect=failing_writer)
    else:
        source = BytesIO(b"content")
        failure = patch.object(
            _POLICY.os, "replace", side_effect=OSError("rename failed")
        )
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with failure, pytest.raises(OSError, match=f"{stage} failed"):
            _POLICY.upload_at(fd, "file", source)
    finally:
        os.close(fd)
    assert destination.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [destination]


def test_cleanup_failure_does_not_mask_the_upload_error(tmp_path, caplog):
    source = Mock(read=Mock(side_effect=OSError("original read failure")))
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with patch.object(_POLICY.os, "unlink", side_effect=OSError("cleanup failed")):
            with pytest.raises(OSError, match="original read failure"):
                _POLICY.upload_at(fd, "file", source)
    finally:
        os.close(fd)
    assert "Could not remove failed upload temporary file" in caplog.text
    assert not (tmp_path / "file").exists()


def test_failed_exclusive_create_does_not_remove_an_unowned_file(tmp_path):
    nonce = b"x" * 16
    unowned = tmp_path / f".faith-upload-{nonce.hex()}"
    unowned.write_bytes(b"keep")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with patch.object(_POLICY.os, "urandom", return_value=nonce):
            with pytest.raises(FileExistsError):
                _POLICY.upload_at(fd, "file", BytesIO(b"content"))
    finally:
        os.close(fd)
    assert unowned.read_bytes() == b"keep"


def test_sealing_waits_for_upload_to_finish_atomic_publication(tmp_path):
    policy = _policy(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    entered, release, sealing = Event(), Event(), Event()

    class PausedStream(BytesIO):
        """Hold copying open while a concurrent seal waits for the policy lock."""

        def read(self, size=-1):
            entered.set()
            assert release.wait(5)
            return super().read(size)

    def upload():
        fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with policy.locked():
                policy.check_upload()
                _POLICY.upload_at(fd, "file", PausedStream(b"complete"))
        finally:
            os.close(fd)

    def seal():
        sealing.set()
        with _policy(tmp_path).locked():
            assert (workspace / "file").read_bytes() == b"complete"
            policy._save({"status": "sealed", "identity": "target:1"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        uploaded = pool.submit(upload)
        try:
            assert entered.wait(2)
            sealed = pool.submit(seal)
            assert sealing.wait(2)
            assert not sealed.done()
            assert not (workspace / "file").exists()
        finally:
            release.set()
        uploaded.result(timeout=5)
        sealed.result(timeout=5)
    with pytest.raises(_POLICY.PolicyError):
        policy.check_upload()
