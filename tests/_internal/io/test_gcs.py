# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from faith._internal.io.gcs import (
    _ensure_trailing_slash,
    gcp_cp,
    gcp_is_file,
    gcp_rsync,
    gcp_url,
)


def test_ensure_trailing_slash() -> None:
    assert _ensure_trailing_slash("gs://bucket/path") == "gs://bucket/path/"
    assert _ensure_trailing_slash("gs://bucket/path/") == "gs://bucket/path/"


def test_gcp_url() -> None:
    assert gcp_url(Path("bucket/path")) == "gs://bucket/path"


def test_gcp_cp_success() -> None:
    with patch("faith._internal.io.gcs.Popen") as mock_popen:
        mock_process = Mock(returncode=0)
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        gcp_cp("gs://bucket/file.txt", "/tmp/file.txt")
        mock_popen.assert_called_once_with(
            ["gsutil", "cp", "-J", "gs://bucket/file.txt", "/tmp/file.txt"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )


def test_gcp_cp_failure_raises() -> None:
    with patch("faith._internal.io.gcs.Popen") as mock_popen:
        mock_process = Mock(returncode=1)
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        with pytest.raises(RuntimeError, match="Failed to run GCP command"):
            gcp_cp("gs://bucket/file.txt", "/tmp/file.txt", raise_on_error=True)


def test_gcp_rsync_success() -> None:
    with patch("faith._internal.io.gcs.Popen") as mock_popen:
        mock_process = Mock(returncode=0)
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        gcp_rsync("gs://bucket/dir", "/tmp/dir")
        mock_popen.assert_called_once_with(
            [
                "gsutil",
                "-m",
                "rsync",
                "-R",
                "-J",
                "-P",
                "gs://bucket/dir/",
                "/tmp/dir/",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )


def test_gcp_rsync_failure_raises() -> None:
    with patch("faith._internal.io.gcs.Popen") as mock_popen:
        mock_process = Mock(returncode=1)
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        with pytest.raises(RuntimeError, match="Failed to run GCP command"):
            gcp_rsync("gs://bucket/dir", "/tmp/dir", raise_on_error=True)


def test_gcp_is_file_true() -> None:
    file_output = """
        1234  2024-01-01T00:00:00Z  gs://test-bucket/fake-datastore/bar.txt
        TOTAL: 1 objects, 1234 bytes (1.21 KiB)
"""
    with patch("faith._internal.io.gcs.Popen") as mock_popen:
        mock_process = Mock(
            communicate=Mock(return_value=(file_output, "")),
            returncode=0,
        )
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        assert gcp_is_file("gs://test-bucket/fake-datastore/bar.txt") is True
        mock_popen.assert_called_once_with(
            ["gsutil", "ls", "-d", "-l", "gs://test-bucket/fake-datastore/bar.txt"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )


def test_gcp_is_file_false() -> None:
    dir_output = "                                 gs://test-bucket/fake-datastore/"
    with patch("faith._internal.io.gcs.Popen") as mock_popen:
        mock_process = Mock(
            communicate=Mock(return_value=(dir_output, "")),
            returncode=0,
        )
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        assert gcp_is_file("gs://test-bucket/fake-datastore") is False


def test_gcp_is_file_failure() -> None:
    with patch("faith._internal.io.gcs.Popen") as mock_popen:
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
            gcp_is_file("gs://test-bucket/bad-path")
