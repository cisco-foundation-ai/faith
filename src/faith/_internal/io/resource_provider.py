# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""ResourceProvider: uniform local access to resources regardless of storage backend."""

import tempfile
from pathlib import Path
from types import TracebackType

from faith._internal.io.gcs import gcp_cp, gcp_is_file, gcp_rsync


class ResourceProvider:
    """Context manager that provides a local path to a file or directory.

    Abstracts over storage backends so callers always get a readable
    local Path, regardless of where the resource actually lives.

    Local paths:  validates existence, returns the Path directly.
    GCS URIs:     downloads to a temp directory, returns the local Path.
    """

    def __init__(self, uri: str):
        self._uri = uri
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> Path:
        if self._uri.startswith("gs://"):
            return self._enter_gcs()
        return self._enter_local()

    def _enter_local(self) -> Path:
        path = Path(self._uri)
        if not path.exists():
            raise FileNotFoundError(f"Path '{self._uri}' does not exist.")
        return path

    def _enter_gcs(self) -> Path:
        # pylint: disable-next=consider-using-with
        self._temp_dir = tempfile.TemporaryDirectory()
        local_dir = Path(self._temp_dir.name)
        if gcp_is_file(self._uri):
            filename = Path(self._uri[5:]).name  # Strip 'gs://' prefix
            gcp_cp(self._uri, str(local_dir), raise_on_error=True)
            return local_dir / filename
        gcp_rsync(self._uri, str(local_dir), raise_on_error=True)
        return local_dir

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
