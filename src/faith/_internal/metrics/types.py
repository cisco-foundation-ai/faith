# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Common types for labels and predictions in evaluation metrics."""

from collections.abc import Sequence

Labeling = str | Sequence[str]
SingleLabelSeq = Sequence[str | None]
MultiLabelSeq = Sequence[Sequence[str] | None]
