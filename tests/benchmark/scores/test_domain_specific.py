# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from cvss.exceptions import CVSS3MalformedError

from faith.benchmark.scores.domain_specific import (
    AliasAccuracyScore,
    CompositeScore,
    CVSSScore,
    DomainSpecificScore,
    JaccardIndex,
    LogScaledScore,
)
from faith.benchmark.scores.types import Score


def test_score_fn_enum() -> None:
    # Test valid score function names
    assert DomainSpecificScore.from_string("cvss") == DomainSpecificScore.CVSS
    assert str(DomainSpecificScore.CVSS) == "cvss"
    assert (
        DomainSpecificScore.CVSS.get_score_fn() is not None
    ), "DomainSpecificScore should return a valid scoring function instance"

    # Test invalid score function name
    with pytest.raises(ValueError):
        DomainSpecificScore.from_string("invalid_score_fn")


def test_score_fn_from_configs() -> None:
    scores = DomainSpecificScore.from_configs(
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
    assert 0 <= score <= 10, "CVSS score should be normalized to [0, 10]"

    # Test invalid CVSS vector
    with pytest.raises(CVSS3MalformedError, match="Malformed CVSS3 vector"):
        score_fn.get_cvss_score("INVALID:VECTOR")


def test_cvssscore() -> None:
    score_fn = CVSSScore()
    label = "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"

    equivalent_pred = "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    score = score_fn(label, equivalent_pred)
    assert score == {
        "value": pytest.approx(1),
        "raw_value": pytest.approx(10),
    }, "Score should be 1.0 for perfect match"

    different_pred = "CVSS:3.0/S:C/C:H/I:H/A:N/AV:P/AC:H/PR:H/UI:R/E:H/RL:O/RC:R/CR:H/IR:X/AR:X/MAC:H/MPR:X/MUI:X/MC:L/MA:X"
    score = score_fn(label, different_pred)
    assert (
        0.0 <= score["value"] < 1.0
    ), "Score should be less than 1.0 for different vectors"

    invalid_pred = "INVALID:VECTOR"
    score = score_fn(label, invalid_pred)
    assert score == {
        "value": pytest.approx(0),
        "raw_value": pytest.approx(0),
    }, "Score should be 0.0 for invalid prediction"

    score = score_fn(label, None)
    assert score == {
        "value": pytest.approx(0),
        "raw_value": pytest.approx(0),
    }, "Score should be 0.0 for None prediction"


def test_cvssscore_aggregate() -> None:
    score_fn = CVSSScore()

    scores = [
        Score(value=0.9, raw_value=9),
        Score(value=0.8, raw_value=8),
        Score(value=0.83, raw_value=8.3),
    ]
    aggregated_scores = score_fn.aggregate(scores)
    assert aggregated_scores == {
        "mean": pytest.approx(0.84333333333),
        "median": pytest.approx(0.83),
    }


def test_jaccard_index() -> None:
    score_fn = JaccardIndex()
    label = ("tag1", "tag2")

    assert score_fn(label, ("tag1", "tag2")) == {
        "value": pytest.approx(1),
        "raw_value": pytest.approx(1),
    }
    assert score_fn(label, ("tag3", "tag4")) == {
        "value": pytest.approx(0),
        "raw_value": pytest.approx(0),
    }
    assert score_fn(label, ("tag2", "tag3")) == {
        "value": pytest.approx(1 / 3),
        "raw_value": pytest.approx(1 / 3),
    }


def test_jaccard_index_aggregate() -> None:
    score_fn = JaccardIndex()

    scores = [
        Score(value=0.1, raw_value=0.1),
        Score(value=0.6, raw_value=0.6),
        Score(value=0.2, raw_value=0.2),
    ]
    aggregated_scores = score_fn.aggregate(scores)
    assert aggregated_scores == {
        "mean": pytest.approx(0.3),
        "median": pytest.approx(0.2),
    }


def test_log_scaled_score() -> None:
    score_fn = LogScaledScore(
        tolerance=0.1, scaling=10.0, score_range={"min": 0, "max": 100}
    )

    assert score_fn("1.0", None) == {
        "value": pytest.approx(0),
        "raw_value": pytest.approx(0),
    }
    assert score_fn("2.0", "I can't answer that") == {
        "value": pytest.approx(0),
        "raw_value": pytest.approx(0),
    }
    assert score_fn("10.0", "10") == {
        "value": pytest.approx(100),
        "raw_value": pytest.approx(1),
    }
    assert score_fn("100", "1.0e+2") == {
        "value": pytest.approx(100),
        "raw_value": pytest.approx(1),
    }
    assert score_fn("10000", "9999") == {
        "value": pytest.approx(99.95831759858649),
        "raw_value": pytest.approx(0.9995831759858649),
    }
    assert score_fn("100", "50") == {
        "value": pytest.approx(25.277826369078593),
        "raw_value": pytest.approx(0.25277826369078593),
    }
    assert score_fn("10000", "-10000") == {
        "value": pytest.approx(0),
        "raw_value": pytest.approx(0),
    }


def test_log_scaled_score_aggregate() -> None:
    score_fn = LogScaledScore(tolerance=0.1, scaling=10.0)

    scores = [
        Score(value=0.1, raw_value=0.1),
        Score(value=0.6, raw_value=0.6),
        Score(value=0.2, raw_value=0.2),
        Score(value=1.0, raw_value=1.0),
    ]
    aggregated_scores = score_fn.aggregate(scores)
    assert aggregated_scores == {
        "mean": pytest.approx(0.475),
    }


def test_alias_accuracy() -> None:
    score_fn = AliasAccuracyScore(
        {
            "actor1": ["alias1", "alias2"],
            "actor2": ["alias3"],
            "actor3": ["alias1", "alias4"],
        },
        score_range={"min": 2, "max": 7},
    )
    assert score_fn("actor1", None) == {
        "value": pytest.approx(2),
        "raw_value": pytest.approx(0),
    }
    assert score_fn("actor1", "alias1") == {
        "value": pytest.approx(7),
        "raw_value": pytest.approx(1),
    }
    assert score_fn("actor1", "actor3") == {
        "value": pytest.approx(7),
        "raw_value": pytest.approx(1),
    }
    assert score_fn("actor2", "actor1") == {
        "value": pytest.approx(2),
        "raw_value": pytest.approx(0),
    }


def test_alias_accuracy_aggregate() -> None:
    score_fn = AliasAccuracyScore(
        {
            "actor1": ["alias1", "alias2"],
            "actor2": ["alias3"],
            "actor3": ["alias1", "alias4"],
        }
    )

    scores = [
        Score(value=1.0, raw_value=1.0),
        Score(value=0.0, raw_value=0.0),
        Score(value=1.0, raw_value=1.0),
    ]
    aggregated_scores = score_fn.aggregate(scores)
    assert aggregated_scores == {"accuracy": pytest.approx(2 / 3)}


def test_composite_score_fn_from_configs() -> None:
    scores = DomainSpecificScore.from_configs(
        weighted_score={
            "type": "composite",
            "aggregation": "0.7 * scores.cvss_score + 0.3 * scores.jaccard_index",
            "cvss_score": {"type": "cvss", "attributes": {"weight": 0.7}},
            "jaccard_index": {"type": "jaccard", "attributes": {"weight": 0.3}},
        }
    )
    assert set(scores.keys()) == {"weighted_score"}
    assert isinstance(scores["weighted_score"], CompositeScore)


def test_composite_score() -> None:
    score_fn = CompositeScore(
        aggregation="0.7 * scores.log_1 + 0.3 * scores.log_2",
        log_1={"type": "log_scaled_score", "tolerance": 0.1, "scaling": 10.0},
        log_2={"type": "log_scaled_score", "tolerance": 0.5, "scaling": 2.0},
    )
    assert score_fn("100", None) == {
        "value": pytest.approx(0),
        "raw_value": pytest.approx(0),
        "sub_scores": {
            "log_1": {"value": pytest.approx(0), "raw_value": pytest.approx(0)},
            "log_2": {"value": pytest.approx(0), "raw_value": pytest.approx(0)},
        },
    }
    assert score_fn("10", "10") == {
        "value": pytest.approx(1),
        "raw_value": pytest.approx(1),
        "sub_scores": {
            "log_1": {"value": pytest.approx(1), "raw_value": pytest.approx(1)},
            "log_2": {"value": pytest.approx(1), "raw_value": pytest.approx(1)},
        },
    }
    assert score_fn("1000", "900") == {
        "value": pytest.approx(0.7 * 0.7109351736821121 + 0.3 * 0.8340437671464698),
        "raw_value": pytest.approx(0.7 * 0.7109351736821121 + 0.3 * 0.8340437671464698),
        "sub_scores": {
            "log_1": {
                "value": pytest.approx(0.7109351736821121),
                "raw_value": pytest.approx(0.7109351736821121),
            },
            "log_2": {
                "value": pytest.approx(0.8340437671464698),
                "raw_value": pytest.approx(0.8340437671464698),
            },
        },
    }

    attr_weight_score_fn = CompositeScore(
        aggregation="sum(scores[k] * attrs[k]['weight'] for k in scores.keys()) / sum(a['weight'] for a in attrs.values())",
        log_1={
            "type": "log_scaled_score",
            "tolerance": 0.1,
            "scaling": 10.0,
            "attributes": {"weight": 3},
        },
        log_2={
            "type": "log_scaled_score",
            "tolerance": 0.5,
            "scaling": 2.0,
            "attributes": {"weight": 1},
        },
    )
    assert attr_weight_score_fn("100", None) == {
        "value": pytest.approx(0),
        "raw_value": pytest.approx(0),
        "sub_scores": {
            "log_1": {"value": pytest.approx(0), "raw_value": pytest.approx(0)},
            "log_2": {"value": pytest.approx(0), "raw_value": pytest.approx(0)},
        },
    }
    assert attr_weight_score_fn("10", "10") == {
        "value": pytest.approx(1),
        "raw_value": pytest.approx(1),
        "sub_scores": {
            "log_1": {"value": pytest.approx(1), "raw_value": pytest.approx(1)},
            "log_2": {"value": pytest.approx(1), "raw_value": pytest.approx(1)},
        },
    }
    assert attr_weight_score_fn("1000", "900") == {
        "value": pytest.approx((3 * 0.7109351736821121 + 0.8340437671464698) / 4),
        "raw_value": pytest.approx((3 * 0.7109351736821121 + 0.8340437671464698) / 4),
        "sub_scores": {
            "log_1": {
                "value": pytest.approx(0.7109351736821121),
                "raw_value": pytest.approx(0.7109351736821121),
            },
            "log_2": {
                "value": pytest.approx(0.8340437671464698),
                "raw_value": pytest.approx(0.8340437671464698),
            },
        },
    }


def test_composite_score_aggregate() -> None:
    score_fn = CompositeScore(
        aggregation="0.7 * scores.log_1 + 0.3 * scores.log_2",
        log_1={"type": "log_scaled_score", "tolerance": 0.1, "scaling": 10.0},
        log_2={"type": "log_scaled_score", "tolerance": 0.5, "scaling": 2.0},
    )

    scores = [
        Score(value=0.2, raw_value=0.2),
        Score(value=0.6, raw_value=0.6),
        Score(value=0.8, raw_value=0.8),
    ]
    aggregated_scores = score_fn.aggregate(scores)
    assert aggregated_scores == {"mean": pytest.approx(8 / 15)}
