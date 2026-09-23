# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Output truncation shared by the tools that can return unbounded text."""


def truncate(output: str, max_chars: int) -> str:
    """Head-truncate with a marker reporting the true total.

    The total is what tells the agent to narrow its search rather than conclude it
    saw everything; a head cap on `grep -r` returns whichever files sort first.
    """
    if len(output) <= max_chars:
        return output
    return (
        output[:max_chars]
        + f"\n\n[TRUNCATED — {len(output)} total chars, showing first {max_chars}]"
    )
