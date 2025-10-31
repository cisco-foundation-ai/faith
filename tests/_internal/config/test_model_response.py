# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._internal.config.model_response import model_response_format_config


def test_model_response_format_config() -> None:
    assert model_response_format_config() == {
        "pattern": r"(?s).*",
        "match_disambiguation": "match_all",
        "format_type": "proper",
    }

    custom_pattern = r"\d+"
    assert model_response_format_config(custom_pattern) == {
        "pattern": custom_pattern,
        "match_disambiguation": "match_all",
        "format_type": "proper",
    }
