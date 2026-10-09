# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Per-benchmark task loaders, one module each.

A loader turns a benchmark's own metadata format into the Tasks the agentic
runner executes (see `AgenticTaskLoaderType`).
"""

from faith.agentic.loaders.vulnerability_localization import (
    VulnerabilityLocalizationTaskLoader,
)

__all__ = [
    "VulnerabilityLocalizationTaskLoader",
]
