# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Store: bidirectional sync for data between local and remote storage."""

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType

from faith._internal.io.gcs import gcp_rsync, gcp_url


class Store(ABC):
    """Bidirectional sync for data between local and remote storage.

    Always directory-based. Supports hierarchical organization via sub_store().
    """

    @property
    @abstractmethod
    def path(self) -> Path:
        """Return the local path to the store."""

    @abstractmethod
    def pull(self, *, raise_on_error: bool = False) -> Path:
        """Synchronize remote data to the local store and return its local path."""

    @abstractmethod
    def push(self, *, raise_on_error: bool = False) -> None:
        """Synchronize local data to the remote store."""

    @abstractmethod
    def sub_store(self, sub_path: Path) -> "Store":
        """Return a sub-store at the given sub_path of the current store."""


class _LocalStore(Store):
    """A local store — pull/push are no-ops."""

    def __init__(self, path: Path):
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def pull(self, *, raise_on_error: bool = False) -> Path:
        return self.path

    def push(self, *, raise_on_error: bool = False) -> None:
        pass

    def sub_store(self, sub_path: Path) -> Store:
        return _LocalStore(self.path / sub_path)


class _GCPStore(Store):
    """A GCP-backed store that uses gsutil rsync for sync."""

    def __init__(self, bucket_path: Path, local_path: Path):
        self._bucket_path = bucket_path
        self._local_path = local_path

    @property
    def path(self) -> Path:
        return self._local_path

    def pull(self, *, raise_on_error: bool = False) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        gcp_rsync(
            gcp_url(self._bucket_path), str(self.path), raise_on_error=raise_on_error
        )
        return self.path

    def push(self, *, raise_on_error: bool = False) -> None:
        assert self.path.exists(), f"Local path {self.path} does not exist."
        gcp_rsync(
            str(self.path), gcp_url(self._bucket_path), raise_on_error=raise_on_error
        )

    def sub_store(self, sub_path: Path) -> Store:
        return _GCPStore(self._bucket_path / sub_path, self.path / sub_path)


class StoreContext(ABC):
    """Context manager for store lifecycle."""

    @staticmethod
    def from_path(path: str) -> "StoreContext":
        """Return an StoreContext for the given path/url."""
        if path.startswith("gs://"):
            return _GCPStoreContext(path)
        return _LocalStoreContext(path)

    @abstractmethod
    def __enter__(self) -> Store:
        """Create and return the store."""

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up any resources."""


class _LocalStoreContext(StoreContext):

    def __init__(self, path: str):
        self._path = Path(path)

    def __enter__(self) -> Store:
        return _LocalStore(self._path)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


class _GCPStoreContext(StoreContext):

    def __init__(self, gcp_addr: str):
        assert gcp_addr.startswith("gs://"), "GCP path must start with 'gs://'."
        self._gcp_addr = gcp_addr
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> Store:
        # pylint: disable-next=consider-using-with
        self._temp_dir = tempfile.TemporaryDirectory()
        bucket_path = Path(self._gcp_addr[5:])  # Strip 'gs://' prefix
        return _GCPStore(bucket_path, Path(self._temp_dir.name))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
