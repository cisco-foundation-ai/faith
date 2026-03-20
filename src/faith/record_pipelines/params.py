# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from faith._types.record.sample_record import ReplacementStrategy


@dataclass(frozen=True)
class RecordHandlingParams:
    """Parameters defining the behavior of record reconciliation and stats computation."""

    replacement_strategy: ReplacementStrategy
