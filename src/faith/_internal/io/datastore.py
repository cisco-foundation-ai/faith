# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for managing datastores and synchronizing them with GCP storage."""
import logging
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from subprocess import DEVNULL, STDOUT, Popen
from types import TracebackType

logger = logging.getLogger(__name__)


class DataStore(ABC):
    """An abstract base class for a datastore."""

    @staticmethod
    def sub_path(path: Path, sub_path: Path | None) -> Path:
        """Return the sub_path of the given path, or the path itself if sub_path is None."""
        return path if sub_path is None else path / sub_path

    @abstractmethod
    def sub_store(self, sub_path: Path) -> "DataStore":
        """Return a sub-store at the given `sub_path` of the current store."""

    @abstractmethod
    def push(self) -> None:
        """Synchronize the local store to the datastore."""

    @abstractmethod
    def pull(self) -> Path:
        """Synchronize the datastore to the local store and return its local path."""


class LocalDataStore(DataStore):
    """A class to represent a local data store."""

    def __init__(self, path: Path):
        """Initialize the LocalDataStore at a given path."""
        self._path = path

    def sub_store(self, sub_path: Path) -> "LocalDataStore":
        """Return a sub-store for the given `sub_path`."""
        local_path = DataStore.sub_path(self._path, sub_path)
        return LocalDataStore(local_path)

    def push(self) -> None:
        """Synchronize the local store to the datastore.

        This is a no-op for local stores, as they are already in sync.
        """

    def pull(self) -> Path:
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
            # pylint: disable=consider-using-with
            self._temp_dir = tempfile.TemporaryDirectory()
            self._local_path = Path(self._temp_dir.name)
        else:
            self._local_path = local_path

    def __del__(self) -> None:
        """Clean up the temporary directory when this object is deleted."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    @property
    def _local_store(self) -> Path:
        """Return the local path where the store is housed."""
        return self._local_path


class GCPDataStore(RemoteDataStore):
    """A GCP-base data store that uses `gsutil rsync` to synchronize data."""

    def __init__(self, remote_path: str, local_path: Path | None = None):
        """Initialize the GCPDataStore with a given remote path/url."""
        super().__init__(local_path)
        assert remote_path.startswith(
            "gs://"
        ), "GCP datastore location must start with 'gs://'."
        self._bucket_path = Path(remote_path[5:])  # Remove 'gs://' prefix

    @staticmethod
    def _rsync_cmd_args(src: str, dest: str) -> list[str]:
        """Return the command arguments to execute rsync from src to dest."""
        return [
            "gsutil",
            "-m",
            "rsync",
            "-R",
            "-J",
            "-P",
            src,
            dest,
        ]

    @staticmethod
    def _rsync(src: str, dest: str) -> None:
        """Execute the rsync command to synchronize src to dest."""
        with Popen(
            GCPDataStore._rsync_cmd_args(src, dest),  # noqa: SLF001
            stdout=sys.stdout,
            stderr=sys.stderr,
        ) as process:
            process.wait()
            if process.returncode != 0:
                raise ValueError(f"Failed to rsync from '{src}' to '{dest}'.")

    def _gcp_url(self, bucket_path: Path) -> str:
        """Return the GCP store url as a string."""
        return f"gs://{str(bucket_path)}"

    def sub_store(self, sub_path: Path) -> DataStore:
        """Return a sub-store for the given `sub_path`."""
        remote_sub_path = DataStore.sub_path(self._bucket_path, sub_path)
        return GCPDataStore(
            self._gcp_url(remote_sub_path), self._local_store / sub_path
        )

    def push(self) -> None:
        """Synchronize the local store to the datastore.

        This will push the contents of the local store to the GCP bucket path.
        """
        assert (
            self._local_store.exists()
        ), f"Local path {self._local_store} does not exist."
        self._rsync(str(self._local_store), self._gcp_url(self._bucket_path))

    def pull(self) -> Path:
        """Synchronize the datastore to the local store and return its local path.

        This will pull the contents of the GCP bucket path to the local store.
        """
        # Ensure the local path exists.
        self._local_store.mkdir(parents=True, exist_ok=True)
        self._rsync(self._gcp_url(self._bucket_path), str(self._local_store))
        return self._local_store


def _datastore_from_path(path: str) -> DataStore:
    """Return a DataStore instance for the given path/url.

    If the path/url starts with "gs://", return a GCPDataStore.
    Otherwise, return a LocalDataStore for the local path.
    """
    if path.startswith("gs://"):
        return GCPDataStore(path)
    else:
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
            del self._datastore  # Clean up the local datastore.


class ReadOnlyDataContext:
    """A context manager for a read-only data store for directories or files."""

    def __init__(self, path_to_data: str, is_file: bool):
        """Initialize the ReadOnlyDataContext with a given path."""
        self._path_to_data = path_to_data
        self._is_file = is_file
        self._datastore: DataStore | None = None

    def __enter__(self) -> Path:
        """Return the path to the data, ensuring it exists."""
        datastore_path = self._path_to_data
        filename = None
        if self._is_file:
            filename = os.path.basename(datastore_path)
            datastore_path = os.path.dirname(datastore_path)
        self._datastore = _datastore_from_path(datastore_path)

        local_path = self._datastore.pull()
        if filename is not None:
            local_path = local_path / filename
        return local_path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up the datastore."""
        if self._datastore is not None:
            del self._datastore


class GCPSynchronizer:
    """A context manager to periodically rsync a local directory to a GCP storage path."""

    def __init__(
        self, gcp_storage_path: str, watch_rate: int = 150, test_run: bool = True
    ):
        """Initialize the GCP synchronizer for the given storage path."""
        # Check preconditions.
        self._gcp_storage_path = gcp_storage_path
        assert self._gcp_storage_path.startswith(
            "gs://"
        ), "GCP storage path must start with 'gs://'."
        self._watch_rate = watch_rate
        assert self._watch_rate > 0, "Watch rate must be a positive."
        # pylint: disable=consider-using-with
        self._temp_dir = tempfile.TemporaryDirectory()
        self._watch_process: Popen | None = None

        # Execute a test run of rsync to ensure it is configured correctly.
        if test_run:
            logger.info("Running gsutil rsync test...")
            with Popen(
                self._rsync_cmd_args(),
                stdout=sys.stdout,
                stderr=sys.stderr,
            ) as trial_upload:
                trial_upload.wait()
                if trial_upload.returncode != 0:
                    raise ValueError(
                        f"Initial upload to GCP failed with return code {trial_upload.returncode}."
                    )
            logger.info("Test of gsutil rsync succeeded.")

    def __del__(self) -> None:
        """Clean up the temporary directory when this object is deleted."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    def _rsync_cmd_args(self) -> list[str]:
        """Return the command arguments to execute rsync to the GCP storage path."""
        return [
            "gsutil",
            "rsync",
            "-R",
            "-J",
            "-P",
            str(self._temp_dir.name),
            str(self._gcp_storage_path),
        ]

    def __enter__(self) -> Path:
        """Create a temporary directory and start periodic rsync to GCP storage."""
        logger.info("Running periodic gsutil rsync...")
        self._watch_process = Popen(
            ["watch", "-n", str(self._watch_rate)] + self._rsync_cmd_args(),
            stdout=DEVNULL,
            stderr=STDOUT,
        )
        return Path(self._temp_dir.name)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up the temporary directory and perform a final rsync."""
        logger.info("Halting periodic gsutil rsync...")
        if self._watch_process:
            self._watch_process.terminate()
            self._watch_process.wait()
        with Popen(self._rsync_cmd_args(), stdout=DEVNULL, stderr=STDOUT) as final_sync:
            final_sync.wait()
            assert final_sync.returncode == 0, "Final gsutil rsync failed."
        self._temp_dir.cleanup()


class _PathContext:
    """A context manager for a path to mirror a GCPSynchronizer."""

    def __init__(self, path: Path):
        """Initialize the context with a given path."""
        self._path = path

    def __enter__(self) -> Path:
        """Return the path for use in a context."""
        return self._path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context without any cleanup."""


def resolve_storage_path(base_dir: str) -> _PathContext | GCPSynchronizer:
    """Resolve the base directory to a local path or a GCP synchronizer."""
    if base_dir.startswith("gs://"):
        # If the base directory is a GCP storage path, use GCPUploader.
        return GCPSynchronizer(base_dir)
    # Otherwise, return the local path.
    return _PathContext(Path(base_dir))
