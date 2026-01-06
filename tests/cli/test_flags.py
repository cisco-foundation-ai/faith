# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from faith.cli.flags import parse_begin_end_tokens


def test_parse_begin_end_tokens() -> None:
    assert parse_begin_end_tokens("start, end") == ("start", "end")
    assert parse_begin_end_tokens("'[0, 1]', '2'") == ("[0, 1]", "2")
    assert parse_begin_end_tokens('"0", "[1, 2]"') == ("0", "[1, 2]")
    assert parse_begin_end_tokens("42, 100") == ([42], [100])
    assert parse_begin_end_tokens("[10], [20]") == ([10], [20])
    assert parse_begin_end_tokens("[1, 2, 3], [4,5, 6 ]") == ([1, 2, 3], [4, 5, 6])

    with pytest.raises(ValueError, match="Invalid token-sequence pair: 'invalid'"):
        parse_begin_end_tokens("invalid")
    with pytest.raises(ValueError, match="Invalid token-sequence pair: '\\[4, 5\\]'"):
        parse_begin_end_tokens("[4, 5]")
