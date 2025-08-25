# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from cvss.exceptions import CVSS3MalformedError

from faith._internal.metrics.domain_specific_scores import (
    AliasAccuracyScore,
    CVSSScore,
    JaccardIndex,
    ScoreFn,
)


def test_score_fn_enum() -> None:
    # Test valid score function names
    assert ScoreFn.from_string("cvss") == ScoreFn.CVSS
    assert str(ScoreFn.CVSS) == "cvss"
    assert (
        ScoreFn.CVSS.get_score_fn() is not None
    ), "ScoreFn should return a valid scoring function instance"

    # Test invalid score function name
    with pytest.raises(ValueError):
        ScoreFn.from_string("invalid_score_fn")


def test_score_fn_from_configs() -> None:
    scores = ScoreFn.from_configs(
        cvss_score={"type": "cvss"},
        aliases={
            "type": "alias_accuracy",
            "alias_map": {
                "Benjamin Franklin": [
                    "Silence Dogood",
                    "Anthony Afterwit",
                    "Benevolus",
                ],
                "Samuel Clemens": ["Mark Twain"],
            },
        },
    )
    assert set(scores.keys()) == {"cvss_score", "aliases"}
    assert isinstance(scores["cvss_score"], CVSSScore)
    assert isinstance(scores["aliases"], AliasAccuracyScore)


def test_get_cvss_score() -> None:
    # Test valid CVSS vector
    cvss_vector = "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    score_fn = CVSSScore()
    score = score_fn.get_cvss_score(cvss_vector)
    assert 0 <= score <= 1, "CVSS score should be normalized to [0, 1]"

    # Test invalid CVSS vector
    with pytest.raises(CVSS3MalformedError, match="Malformed CVSS3 vector"):
        score_fn.get_cvss_score("INVALID:VECTOR")


def test_cvssscore() -> None:
    score_fn = CVSSScore()
    label = "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"

    equivalent_pred = "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    score = score_fn(label, equivalent_pred)
    assert score == 1.0, "Score should be 1.0 for perfect match"

    different_pred = "CVSS:3.0/S:C/C:H/I:H/A:N/AV:P/AC:H/PR:H/UI:R/E:H/RL:O/RC:R/CR:H/IR:X/AR:X/MAC:H/MPR:X/MUI:X/MC:L/MA:X"
    score = score_fn(label, different_pred)
    assert 0.0 <= score < 1.0, "Score should be less than 1.0 for different vectors"

    invalid_pred = "INVALID:VECTOR"
    score = score_fn(label, invalid_pred)
    assert score == 0.0, "Score should be 0.0 for invalid prediction"

    score = score_fn(label, None)
    assert score == 0.0, "Score should be 0.0 for None prediction"


def test_cvssscore_aggregate() -> None:
    score_fn = CVSSScore()

    scores = [0.9, 0.8, 0.83]
    aggregated_scores = score_fn.aggregate(scores)
    assert aggregated_scores == {
        "mean": pytest.approx(0.84333333333),
        "median": pytest.approx(0.83),
    }


def test_jaccard_index() -> None:
    score_fn = JaccardIndex()
    label = ("tag1", "tag2")

    assert score_fn(label, ("tag1", "tag2")) == pytest.approx(1.0)
    assert score_fn(label, ("tag3", "tag4")) == pytest.approx(0.0)
    assert score_fn(label, ("tag2", "tag3")) == pytest.approx(0.33333333333)


def test_jaccard_index_aggregate() -> None:
    score_fn = JaccardIndex()

    scores = [0.1, 0.6, 0.2]
    aggregated_scores = score_fn.aggregate(scores)
    assert aggregated_scores == {
        "mean": pytest.approx(0.3),
        "median": pytest.approx(0.2),
    }


def test_alias_accuracy() -> None:
    score_fn = AliasAccuracyScore(
        {
            "actor1": ["alias1", "alias2"],
            "actor2": ["alias3"],
            "actor3": ["alias1", "alias4"],
        }
    )
    assert score_fn("actor1", "alias1") == pytest.approx(1.0)
    assert score_fn("actor1", "actor3") == pytest.approx(1.0)
    assert score_fn("actor2", "actor1") == pytest.approx(0.0)


def test_alias_accuracy_aggregate() -> None:
    score_fn = AliasAccuracyScore(
        {
            "actor1": ["alias1", "alias2"],
            "actor2": ["alias3"],
            "actor3": ["alias1", "alias4"],
        }
    )

    scores = [1.0, 0.0, 1.0]
    aggregated_scores = score_fn.aggregate(scores)
    assert aggregated_scores == {"accuracy": pytest.approx(0.66666666667)}
