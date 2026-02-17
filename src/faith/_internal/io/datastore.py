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


def _cp_cmd_args(src: str, dest: str) -> list[str]:
    """Return the command arguments to execute cp from src to dest."""
    return ["gsutil", "cp", "-J", src, dest]


def _rsync_cmd_args(src: str, dest: str) -> list[str]:
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


class DataStore(ABC):
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
    def sub_store(self, sub_path: Path) -> "DataStore":
        """Return a sub-store at the given `sub_path` of the current store."""

    @abstractmethod
    def push(self, *, raise_on_error: bool = False) -> None:
        """Synchronize the local store to the datastore."""

    @abstractmethod
    def pull(self, *, raise_on_error: bool = False) -> Path:
        """Synchronize the datastore to the local store and return its local path."""


class LocalDataStore(DataStore):
    """A class to represent a local data store."""

    def __init__(self, path: Path):
        """Initialize the LocalDataStore at a given path."""
        self._path = path

    @property
    def path(self) -> Path:
        """Return the path to the local datastore."""
        return self._path

    def sub_store(self, sub_path: Path) -> "LocalDataStore":
        """Return a sub-store for the given `sub_path`."""
        local_path = DataStore.sub_path(self._path, sub_path)
        return LocalDataStore(local_path)

    def push(self, *, raise_on_error: bool = False) -> None:
        """Synchronize the local store to the datastore.

        This is a no-op for local stores, as they are already in sync.
        """

    def pull(self, *, raise_on_error: bool = False) -> Path:
        """Synchronize the datastore to the local store and return its local path.

        This is a no-op for local stores, as they are already in sync.
        """
        return self._path


class RemoteDataStore(DataStore):
    """An abstract base class for a remote data store."""

    def __init__(self, local_path: Path | None = None):
        """Initialize the RemoteDataStore with local temporary housing for the store.

        If local_path is None, a temporary directory will be created to house the store.
        Otherwise, the provided local_path will be used.
        """
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        if local_path is None:
            # If no local path is provided, create a temporary directory.
            # pylint: disable-next=consider-using-with
            self._temp_dir = tempfile.TemporaryDirectory()
            self._local_path = Path(self._temp_dir.name)
        else:
            self._local_path = local_path

    def __del__(self) -> None:
        """Clean up the temporary directory when this object is deleted."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    @property
    def path(self) -> Path:
        """Return the path to the datastore."""
        return self._local_path


def _gcp_url(bucket_path: Path) -> str:
    """Return the GCP store url as a string."""
    return f"gs://{str(bucket_path)}"


class GCPDataStore(RemoteDataStore):
    """A GCP-base data store that uses `gsutil rsync` to synchronize data."""

    def __init__(self, remote_path: str, local_path: Path | None = None):
        """Initialize the GCPDataStore with a given remote path/url."""
        assert remote_path.startswith(
            "gs://"
        ), "GCP datastore location must start with 'gs://'."
        super().__init__(local_path)
        _bucket_path = Path(remote_path[5:])  # Remove 'gs://' prefix
        self._filename = _bucket_path.name if self._is_file(remote_path) else None
        self._bucket_path = _bucket_path.parent if self._filename else _bucket_path

    @staticmethod
    def _rsync(src: str, dest: str, *, raise_on_error: bool = False) -> None:
        """Execute the gsutil rsync command to synchronize src to dest."""
        with Popen(
            _rsync_cmd_args(src, dest),
            stdout=DEVNULL,
            stderr=STDOUT,
            text=True,
        ) as process:
            process.wait()
            if process.returncode != 0:
                if raise_on_error:
                    raise RuntimeError(f"Failed to rsync from '{src}' to '{dest}'.")
                logger.error(
                    "Failed to rsync from '%s' to '%s' with return code %d.",
                    src,
                    dest,
                    process.returncode,
                )

    @staticmethod
    def _cp(src: str, dest: str, *, raise_on_error: bool = False) -> None:
        """Execute the gsutil cp command to copy src to dest."""
        with Popen(
            _cp_cmd_args(src, dest),
            stdout=DEVNULL,
            stderr=STDOUT,
            text=True,
        ) as process:
            process.wait()
            if process.returncode != 0:
                if raise_on_error:
                    raise RuntimeError(f"Failed to copy from '{src}' to '{dest}'.")
                logger.error(
                    "Failed to copy from '%s' to '%s' with return code %d.",
                    src,
                    dest,
                    process.returncode,
                )

    @staticmethod
    def _is_file(gcp_path: str) -> bool:
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

    def sub_store(self, sub_path: Path) -> DataStore:
        """Return a sub-store for the given `sub_path`."""
        remote_sub_path = DataStore.sub_path(self._bucket_path, sub_path)
        return GCPDataStore(_gcp_url(remote_sub_path), self.path / sub_path)

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
            self._cp(
                str(local_file),
                _gcp_url(self._bucket_path / self._filename),
                raise_on_error=raise_on_error,
            )
        else:
            # Otherwise use rsync.
            self._rsync(
                str(self.path),
                _gcp_url(self._bucket_path),
                raise_on_error=raise_on_error,
            )

    def pull(self, *, raise_on_error: bool = False) -> Path:
        """Synchronize the datastore to the local store and return its local path.

        This will pull the contents of the GCP bucket path to the local store.
        """
        # Ensure the local path exists.
        self.path.mkdir(parents=True, exist_ok=True)
        if self._filename:
            # If the bucket path is a file, use cp to copy to sync to the local path.
            self._cp(
                _gcp_url(self._bucket_path / self._filename),
                str(self.path),
                raise_on_error=raise_on_error,
            )
            return self.path / self._filename
        # Otherwise use rsync.
        self._rsync(
            _gcp_url(self._bucket_path), str(self.path), raise_on_error=raise_on_error
        )
        return self.path


def _datastore_from_path(path: str) -> DataStore:
    """Return a DataStore instance for the given path/url.

    If the path/url starts with "gs://", return a GCPDataStore.
    Otherwise, return a LocalDataStore for the local path.
    """
    if path.startswith("gs://"):
        return GCPDataStore(path)
    return LocalDataStore(Path(path))


class DataStoreContext:
    """A context manager for a DataStore."""

    def __init__(self, path: str):
        """Initialize the DataStoreContext with a given path."""
        self._path = path
        self._datastore: DataStore | None = None

    def __enter__(self) -> DataStore:
        """Create and return the appropriate DataStore for the context."""
        self._datastore = _datastore_from_path(self._path)
        return self._datastore

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up the DataStore and perform any necessary finalization."""
        if self._datastore is not None:
            del self._datastore
