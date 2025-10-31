# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-label classification metrics."""

from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score

from faith._internal.metrics.types import MultiLabelSeq


def _tags_to_matrix(
    tags: Sequence[str] | None, tag_to_idx: dict[str, int]
) -> np.ndarray:
    """Convert a list of tags to a binary matrix representation."""
    ret = np.zeros(len(tag_to_idx), dtype=int)
    if tags:
        ret[[tag_to_idx[tag] for tag in tags]] = 1
    return ret


def micro_f1_score(label: MultiLabelSeq, prediction: MultiLabelSeq) -> float:
    """Compute the micro F1 score for multi-label classification."""
    assert len(label) == len(
        prediction
    ), "Label and prediction lists must have the same length."
    if len(label) == 0:
        return float("nan")
    all_tags = sorted(
        {
            tag
            for tags in (list(label) + list(prediction))
            if tags is not None
            for tag in tags
        }
    )
    tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}

    label_mx = np.stack([_tags_to_matrix(tags, tag_to_idx) for tags in label])
    prediction_mx = np.stack([_tags_to_matrix(tags, tag_to_idx) for tags in prediction])
    return f1_score(label_mx, prediction_mx, average="micro")
