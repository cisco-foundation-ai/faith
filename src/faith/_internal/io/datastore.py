# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for managing datastores and synchronizing them with GCP storage."""

import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from subprocess import DEVNULL, PIPE, STDOUT, Popen
from types import TracebackType

logger = logging.getLogger(__name__)


def _ensure_trailing_slash(path: str) -> str:
    """Ensure a directory path ends with a trailing slash.

    This prevents rsync from creating a subdirectory at the destination
    instead of copying the source directory's contents directly into the destination.
    """
    return path if path.endswith("/") else path + "/"


def _gcp_cp_args(src: str, dest: str) -> list[str]:
    """Return the command arguments to execute cp from src to dest."""
    return ["gsutil", "cp", "-J", src, dest]


def _gcp_cp(src: str, dest: str, *, raise_on_error: bool = False) -> None:
    """Execute the gsutil cp command to copy src to dest."""
    _run_gcp_cmd(_gcp_cp_args(src, dest), raise_on_error=raise_on_error)


def _gcp_is_file(gcp_path: str) -> bool:
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


def _gcp_rsync_args(src: str, dest: str) -> list[str]:
    """Return the command arguments to execute rsync from src to dest."""
    return [
        "gsutil",
        "-m",
        "rsync",
        "-R",
        "-J",
        "-P",
        _ensure_trailing_slash(src),
        _ensure_trailing_slash(dest),
    ]


def _gcp_rsync(src: str, dest: str, *, raise_on_error: bool = False) -> None:
    """Execute the gsutil rsync command to synchronize src to dest."""
    _run_gcp_cmd(_gcp_rsync_args(src, dest), raise_on_error=raise_on_error)


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


def _gcp_url(bucket_path: Path) -> str:
    """Return the GCP store url as a string."""
    return f"gs://{str(bucket_path)}"


class Datastore(ABC):
    """An abstract base class for a datastore."""

    @staticmethod
    def sub_path(path: Path, sub_path: Path | None) -> Path:
        """Return the sub_path of the given path, or the path itself if sub_path is None."""
        return path if sub_path is None else path / sub_path

    @property
    @abstractmethod
    def path(self) -> Path:
        """Return the path to the datastore."""

    @abstractmethod
    def pull(self, *, raise_on_error: bool = False) -> Path:
        """Synchronize the datastore to the local store and return its local path."""

    @abstractmethod
    def push(self, *, raise_on_error: bool = False) -> None:
        """Synchronize the local store to the datastore."""

    @abstractmethod
    def sub_store(self, sub_path: Path) -> "Datastore":
        """Return a sub-store at the given `sub_path` of the current store."""


class _LocalDatastore(Datastore):
    """A class to represent a local datastore."""

    def __init__(self, path: Path):
        """Initialize the local datastore at a given path."""
        self._path = path

    @property
    def path(self) -> Path:
        """Return the path to the local datastore."""
        return self._path

    def pull(self, *, raise_on_error: bool = False) -> Path:
        """Synchronize the datastore to the local store and return its local path.

        This is a no-op for local stores, as they are already in sync.
        """
        return self.path

    def push(self, *, raise_on_error: bool = False) -> None:
        """Synchronize the local store to the datastore.

        This is a no-op for local stores, as they are already in sync.
        """

    def sub_store(self, sub_path: Path) -> Datastore:
        """Return a sub-store for the given `sub_path`."""
        local_path = Datastore.sub_path(self.path, sub_path)
        return _LocalDatastore(local_path)


class _GCPDatastore(Datastore):
    """A GCP-based data store that uses `gsutil rsync` to synchronize data."""

    def __init__(self, remote_path: str, local_path: Path):
        """Initialize the datastore with a given remote path/url."""
        assert remote_path.startswith(
            "gs://"
        ), "GCP datastore location must start with 'gs://'."
        self._local_path = local_path
        _bucket_path = Path(remote_path[5:])  # Remove 'gs://' prefix
        self._filename = _bucket_path.name if _gcp_is_file(remote_path) else None
        self._bucket_path = _bucket_path.parent if self._filename else _bucket_path

    @property
    def path(self) -> Path:
        """Return the path to the datastore."""
        return self._local_path

    def pull(self, *, raise_on_error: bool = False) -> Path:
        """Synchronize the datastore to the local store and return its local path.

        This will pull the contents of the GCP bucket path to the local store.
        """
        # Ensure the local path exists.
        self.path.mkdir(parents=True, exist_ok=True)
        if self._filename:
            # If the bucket path is a file, use cp to copy to sync to the local path.
            _gcp_cp(
                _gcp_url(self._bucket_path / self._filename),
                str(self.path),
                raise_on_error=raise_on_error,
            )
            return self.path / self._filename
        # Otherwise use rsync.
        _gcp_rsync(
            _gcp_url(self._bucket_path), str(self.path), raise_on_error=raise_on_error
        )
        return self.path

    def push(self, *, raise_on_error: bool = False) -> None:
        """Synchronize the local store to the datastore.

        This will push the contents of the local store to the GCP bucket path.
        """
        assert self.path.exists(), f"Local path {self.path} does not exist."
        if self._filename:
            # If the bucket path is a file, use cp to copy to sync the bucket path.
            local_file = self.path / self._filename
            assert (
                local_file.exists()
            ), f"Expected local file {local_file} does not exist."
            _gcp_cp(
                str(local_file),
                _gcp_url(self._bucket_path / self._filename),
                raise_on_error=raise_on_error,
            )
        else:
            # Otherwise use rsync.
            _gcp_rsync(
                str(self.path),
                _gcp_url(self._bucket_path),
                raise_on_error=raise_on_error,
            )

    def sub_store(self, sub_path: Path) -> Datastore:
        """Return a sub-store for the given `sub_path`."""
        remote_sub_path = Datastore.sub_path(self._bucket_path, sub_path)
        return _GCPDatastore(_gcp_url(remote_sub_path), self.path / sub_path)


class DatastoreContext(ABC):
    """An abstract base class for a datastore context manager."""

    @staticmethod
    def from_path(path: str) -> "DatastoreContext":
        """Return a DatastoreContext instance for the given path/url."""
        if path.startswith("gs://"):
            return _GCPDatastoreContext(path)
        return _LocalDatastoreContext(path)

    @abstractmethod
    def __enter__(self) -> Datastore:
        """Create and return the appropriate datastore for the context."""

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up any resources used by the context."""


class _LocalDatastoreContext(DatastoreContext):
    """A context manager for managing the resources for a local datastore."""

    def __init__(self, path: str):
        """Initialize the _LocalDatastoreContext with a given path."""
        self._path = Path(path)

    def __enter__(self) -> Datastore:
        """Create and return a local datastore for the context."""
        return _LocalDatastore(self._path)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """No resources to clean up for a local datastore."""


class _RemoteDatastoreContext(DatastoreContext):
    """A context manager for managing the resources for a remote datastore."""

    def __init__(self):
        """Initialize the _RemoteDatastoreContext with a local temporary store."""
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> Datastore:
        """Create and return a remote datastore for the context."""
        # pylint: disable-next=consider-using-with
        self._temp_dir = tempfile.TemporaryDirectory()
        return self.create(Path(self._temp_dir.name))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up the temporary directory used to mirror the remote datastore."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    @abstractmethod
    def create(self, local_path: Path) -> Datastore:
        """Create and return the appropriate remote datastore for the local path."""


class _GCPDatastoreContext(_RemoteDatastoreContext):
    """A context manager for managing remote GCP datastores."""

    def __init__(self, gcp_addr: str):
        """Initialize the _GCPDatastoreContext with a given gcp bucket address."""
        assert gcp_addr.startswith("gs://"), "GCP path must start with 'gs://'."
        super().__init__()
        self._gcp_addr = gcp_addr

    def create(self, local_path: Path) -> Datastore:
        """Create and return a GCP datastore for its gcp bucket address."""
        return _GCPDatastore(self._gcp_addr, local_path)
