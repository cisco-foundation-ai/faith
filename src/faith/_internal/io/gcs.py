# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for interacting with Google Cloud Storage via gsutil."""

import logging
from pathlib import Path
from subprocess import DEVNULL, PIPE, STDOUT, Popen

logger = logging.getLogger(__name__)


def _ensure_trailing_slash(path: str) -> str:
    """Ensure a directory path ends with a trailing slash.

    This prevents rsync from creating a subdirectory at the destination
    instead of copying the source directory's contents directly into the destination.
    """
    return path if path.endswith("/") else path + "/"


def gcp_cp(src: str, dest: str, *, raise_on_error: bool = False) -> None:
    """Execute the gsutil cp command to copy src to dest."""
    _run_gcp_cmd(["gsutil", "cp", "-J", src, dest], raise_on_error=raise_on_error)


def gcp_rsync(src: str, dest: str, *, raise_on_error: bool = False) -> None:
    """Execute the gsutil rsync command to synchronize src to dest."""
    _run_gcp_cmd(
        [
            "gsutil",
            "-m",
            "rsync",
            "-R",
            "-J",
            "-P",
            _ensure_trailing_slash(src),
            _ensure_trailing_slash(dest),
        ],
        raise_on_error=raise_on_error,
    )


def gcp_is_file(gcp_path: str) -> bool:
    """Return True if the GCP path refers to an object (file).

    Uses `gsutil ls -d -l` which outputs a `TOTAL:` summary line
    only when the path matches an actual object, not a prefix.
    """
    with Popen(
        ["gsutil", "ls", "-d", "-l", gcp_path],
        stdout=PIPE,
        stderr=DEVNULL,
        text=True,
    ) as process:
        stdout, _ = process.communicate()
        if process.returncode != 0:
            raise ValueError(f"Failed to check if GCP path '{gcp_path}' is a file.")
        return "TOTAL: 1 objects," in stdout


def gcp_url(bucket_path: Path) -> str:
    """Return the GCP store url as a string."""
    return f"gs://{str(bucket_path)}"


def _run_gcp_cmd(cmd_args: list[str], *, raise_on_error: bool = False) -> None:
    """Execute a gsutil command with the given arguments."""
    with Popen(cmd_args, stdout=DEVNULL, stderr=STDOUT, text=True) as process:
        process.wait()
        if process.returncode != 0:
            if raise_on_error:
                raise RuntimeError(f"Failed to run GCP command: {' '.join(cmd_args)}.")
            logger.error(
                "Failed to run GCP command `%s` with return code %d.",
                " ".join(cmd_args),
                process.returncode,
            )
