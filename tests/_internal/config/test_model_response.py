# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._internal.config.model_response import model_response_format_config
from faith._types.config.patterns import AnswerFormat, Disambiguation, PatternDef


def test_model_response_format_config() -> None:
    assert model_response_format_config() is None

    custom_pattern = r"\d+"
    assert model_response_format_config(custom_pattern) == PatternDef(
        pattern=custom_pattern,
        disambiguation=Disambiguation.MATCH_ALL,
        format_type=AnswerFormat.PROPER,
    )
