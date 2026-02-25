# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith.cli.flags import parse_begin_end_tokens
from faith.model.base import ReasoningSpec


def test_parse_begin_end_tokens() -> None:
    assert parse_begin_end_tokens("start, end") == ReasoningSpec("start", "end")
    assert parse_begin_end_tokens("'[0, 1]', '2'") == ReasoningSpec("[0, 1]", "2")
    assert parse_begin_end_tokens('"0", "[1, 2]"') == ReasoningSpec("0", "[1, 2]")
    assert parse_begin_end_tokens("42, 100") == ReasoningSpec([42], [100])
    assert parse_begin_end_tokens("[10], [20]") == ReasoningSpec([10], [20])
    assert parse_begin_end_tokens("[1, 2, 3], [4,5, 6 ]") == ReasoningSpec(
        [1, 2, 3], [4, 5, 6]
    )

    with pytest.raises(ValueError, match="Invalid token-sequence pair: 'invalid'"):
        parse_begin_end_tokens("invalid")
    with pytest.raises(ValueError, match="Invalid token-sequence pair: '\\[4, 5\\]'"):
        parse_begin_end_tokens("[4, 5]")
