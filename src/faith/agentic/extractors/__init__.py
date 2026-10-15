# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Per-benchmark answer extractors, one module each.

An extractor turns whatever a benchmark's terminal tools submit into a
prediction its score_fns consume (see `AgenticAnswerExtractorType`).
"""

from faith.agentic.extractors.vulnerability_localization import (
    VulnerabilityLocalizationAnswerExtractor,
)

__all__ = [
    "VulnerabilityLocalizationAnswerExtractor",
]
