# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Configurations for model response handling."""

from typing import Any


def model_response_format_config(format_pattern: str | None = None) -> dict[str, Any]:
    """Get the configuration for model response formatting."""
    return {
        "pattern": format_pattern or r"(?s).*",
        "match_disambiguation": "match_all",
        "format_type": "proper",
    }
