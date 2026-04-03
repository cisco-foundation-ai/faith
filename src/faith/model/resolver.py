# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""ResolvedModelPath: resolves a ModelSpec to a usable local name-or-path."""

from types import TracebackType

from faith._internal.io.resource_provider import ResourceProvider
from faith._types.model.spec import ModelSpec


class ResolvedModelPath:
    """Context manager that resolves a ModelSpec to a usable name-or-path.

    HuggingFace identifiers: returned as-is (e.g. "meta-llama/Llama-2-7b")
    Local paths:             returned as-is
    GCS URIs:                downloaded via ResourceProvider, local path returned
    """

    def __init__(self, model_spec: ModelSpec):
        self._spec = model_spec
        self._provider: ResourceProvider | None = None

    def __enter__(self) -> str:
        if self._spec.is_remote:
            self._provider = ResourceProvider(self._spec.path)
            return str(self._provider.__enter__())
        return self._spec.path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._provider is not None:
            self._provider.__exit__(exc_type, exc_val, exc_tb)
