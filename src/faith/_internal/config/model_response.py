# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Configurations for model response handling."""

from faith._types.config.patterns import AnswerFormat, Disambiguation, PatternDef


def model_response_format_config(
    format_pattern: str | None = None,
) -> PatternDef | None:
    """Get the configuration for model response formatting."""
    return (
        PatternDef(
            pattern=format_pattern or r"(?s).*",
            disambiguation=Disambiguation.MATCH_ALL,
            format_type=AnswerFormat.PROPER,
        )
        if format_pattern is not None
        else None
    )
