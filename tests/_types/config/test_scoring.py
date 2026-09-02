# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for output processing and scoring config types."""

import pytest

from faith._types.config.scoring import OutputProcessingConfig, ScoreFnConfig

_JACCARD = ScoreFnConfig(type="jaccard")


def test_an_undotted_score_fn_name_is_addressable_by_primary_metric() -> None:
    assert OutputProcessingConfig.from_dict(
        {
            "primary_metric": "file_f1.mean",
            "score_fns": {"file_f1": {"type": "jaccard"}},
        }
    ) == OutputProcessingConfig(
        primary_metric="file_f1.mean", score_fns={"file_f1": _JACCARD}
    )


@pytest.mark.parametrize(
    "score_fns,named",
    [
        # "file.f1.mean" is ambiguous: primary_metric cannot say which part is
        # the statistic.
        ({"file.f1": _JACCARD}, "file.f1"),
        # Every offending name is reported, sorted, so one parse fixes the yaml
        # rather than one name per attempt.
        ({"c.d": _JACCARD, "ok": _JACCARD, "a.b": _JACCARD}, "a.b, c.d"),
    ],
    ids=["one", "several_sorted"],
)
def test_a_dotted_score_fn_name_is_rejected(score_fns: dict, named: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        OutputProcessingConfig(score_fns=score_fns)

    assert str(excinfo.value) == (
        f"Score function names must not contain '.': {named}. primary_metric "
        f"addresses a statistic as '<score_fn>.<stat>', so a dotted name cannot "
        f"be referenced."
    )
