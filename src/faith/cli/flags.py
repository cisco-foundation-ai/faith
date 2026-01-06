# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import re

_INT_PAT = r"(?:0|-?[1-9]\d*)"
_INT_CSV_PAT = rf"{_INT_PAT}(?:\s*,\s*{_INT_PAT})*"
_SINGLE_QUOTE_STR_PAT = r"[^']*"
_DOUBLE_QUOTE_STR_PAT = r'[^"]*'
_TOKEN_SEQ_PAT = rf"""
    \[\s*{_INT_CSV_PAT}\s*\]|   # A list of integers
    {_INT_PAT}|                 # or a single integer
    '{_SINGLE_QUOTE_STR_PAT}'|  # or a single-quoted string
    "{_DOUBLE_QUOTE_STR_PAT}"|  # or a double-quoted string
    [^',\[\]]+                  # or an unquoted string
"""


def _parse_tokens(s: str) -> str | list[int]:
    """Parse a token specification string into the appropriate type."""
    if list_matcher := re.fullmatch(rf"\[\s*({_INT_CSV_PAT})\s*\]", s):
        return [int(token.strip()) for token in list_matcher[1].split(",")]
    if int_matcher := re.fullmatch(rf"({_INT_PAT})", s):
        return [int(int_matcher[1].strip())]
    if single_quote_matcher := re.fullmatch(rf"'({_SINGLE_QUOTE_STR_PAT})'", s):
        return single_quote_matcher[1]
    if double_quote_matcher := re.fullmatch(rf'"({_DOUBLE_QUOTE_STR_PAT})"', s):
        return double_quote_matcher[1]
    return s


def parse_begin_end_tokens(s: str) -> tuple[str | list[int], str | list[int]]:
    """Parse a flag string specifying a pair of begin and end reasoning tokens."""
    pair_matcher = re.fullmatch(
        rf"""
            \s*({_TOKEN_SEQ_PAT})\s*
            ,
            \s*({_TOKEN_SEQ_PAT})\s*
        """,
        s,
        re.VERBOSE,
    )
    if not pair_matcher:
        raise ValueError(f"Invalid token-sequence pair: '{s}'")
    return _parse_tokens(pair_matcher[1]), _parse_tokens(pair_matcher[2])
