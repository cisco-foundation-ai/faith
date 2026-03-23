# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from cvss.exceptions import CVSS3MalformedError

from faith._types.config.patterns import AnswerFormat, CaptureTransform, PatternDef
from faith._types.config.scoring import ScoreFnConfig
from faith.benchmark.scores.domain_specific import (
    AliasAccuracyScore,
    CompositeScore,
    CVSSScore,
    DomainSpecificScore,
    JaccardIndex,
    LogScaledScore,
    _decode_by_hint,
    _decode_kwargs_by_hints,
)
from faith.benchmark.scores.scoring import Score

_PATTERN_DICT = {"pattern": ".*", "format_type": "proper"}
_PATTERN_DEF = PatternDef(pattern=".*", format_type=AnswerFormat.PROPER)
_SCORE_FN_DICT = {"type": "cvss"}
_SCORE_FN_CFG = ScoreFnConfig(type="cvss")


class _StubClass:
    def __init__(self, patterns: list[PatternDef], name: str, count: int = 0) -> None:
        pass


@pytest.mark.parametrize(
    "hint, value, expected",
    [
        # Direct DataClassJsonMixin subclass
        (PatternDef, _PATTERN_DICT, _PATTERN_DEF),
        (ScoreFnConfig, _SCORE_FN_DICT, _SCORE_FN_CFG),
        # Already decoded — returned as-is
        (PatternDef, _PATTERN_DEF, _PATTERN_DEF),
        # Primitives — unchanged
        (str, "hello", "hello"),
        (int, 42, 42),
        (float, 3.14, 3.14),
        # list[DataClassJsonMixin]
        (list[PatternDef], [_PATTERN_DICT], [_PATTERN_DEF]),
        # list[str] — no decoding
        (list[str], ["a", "b"], ["a", "b"]),
        # dict[str, DataClassJsonMixin]
        (dict[str, ScoreFnConfig], {"k": _SCORE_FN_DICT}, {"k": _SCORE_FN_CFG}),
        # dict[str, float] — no decoding
        (dict[str, float], {"a": 1.0}, {"a": 1.0}),
        # dict[str, Any] — no decoding (Any must not match DataClassJsonMixin)
        (dict[str, Any], {"a": {"nested": True}}, {"a": {"nested": True}}),
        # Union with DataClassJsonMixin (both orderings)
        (PatternDef | dict[str, Any], _PATTERN_DICT, _PATTERN_DEF),
        (dict[str, Any] | PatternDef, _PATTERN_DICT, _PATTERN_DEF),
        # Optional wrapping (dict[str, ScoreFnConfig] | None)
        (dict[str, ScoreFnConfig] | None, {"k": _SCORE_FN_DICT}, {"k": _SCORE_FN_CFG}),
        (dict[str, ScoreFnConfig] | None, None, None),
        # list of union
        (list[PatternDef | dict[str, Any]], [_PATTERN_DICT], [_PATTERN_DEF]),
        # Empty containers
        (list[PatternDef], [], []),
        (dict[str, ScoreFnConfig], {}, {}),
    ],
    ids=[
        "direct-dataclass",
        "direct-scorefnconfig",
        "already-decoded",
        "str-passthrough",
        "int-passthrough",
        "float-passthrough",
        "list-dataclass",
        "list-str-noop",
        "dict-dataclass",
        "dict-float-noop",
        "dict-any-noop",
        "union-dataclass-first",
        "union-dataclass-second",
        "optional-dict-present",
        "optional-dict-none",
        "list-union-dataclass",
        "empty-list",
        "empty-dict",
    ],
)
def test_decode_by_hint(hint: type, value: Any, expected: Any) -> None:
    assert _decode_by_hint(hint, value) == expected


def test_decode_by_hint_nested_capture_transform() -> None:
    raw = {
        "pattern": r"\b(\d+)\b",
        "format_type": "proper",
        "capture_transform": {"params": ["x"], "expr": "int(x)"},
    }
    result = _decode_by_hint(PatternDef, raw)
    assert isinstance(result, PatternDef)
    assert result.capture_transform == CaptureTransform(params=["x"], expr="int(x)")


def test_decode_kwargs_by_hints_mixed() -> None:
    result = _decode_kwargs_by_hints(
        _StubClass, {"patterns": [_PATTERN_DICT], "name": "test", "count": 5}
    )
    assert result == {"patterns": [_PATTERN_DEF], "name": "test", "count": 5}


def test_decode_kwargs_by_hints_unknown_kwarg_passthrough() -> None:
    result = _decode_kwargs_by_hints(_StubClass, {"unknown": {"foo": "bar"}})
    assert result == {"unknown": {"foo": "bar"}}


def test_domain_specific_score_get_score_fn() -> None:
    assert (
        DomainSpecificScore.CVSS.get_score_fn() is not None
    ), "DomainSpecificScore should return a valid scoring function instance"


def test_domain_specific_score_from_configs() -> None:
    scores = DomainSpecificScore.from_configs(
        cvss_score=ScoreFnConfig(type="cvss"),
        aliases=ScoreFnConfig(
            type="alias_accuracy",
            kwargs={
                "alias_map": {
                    "Benjamin Franklin": [
                        "Silence Dogood",
                        "Anthony Afterwit",
                        "Benevolus",
                    ],
                    "Samuel Clemens": ["Mark Twain"],
                },
            },
        ),
    )
    assert set(scores.keys()) == {"cvss_score", "aliases"}
    assert isinstance(scores["cvss_score"], CVSSScore)
    assert isinstance(scores["aliases"], AliasAccuracyScore)


def test_domain_specific_score_from_configs_with_raw_dict_sub_scores() -> None:
    scores = DomainSpecificScore.from_configs(
        weighted=ScoreFnConfig(
            type="composite",
            kwargs={
                "aggregation": "sub_scores.score.s1",
                "sub_scores": {
                    "s1": {
                        "type": "log_scaled_score",
                        "tolerance": 0.1,
                        "scaling": 10.0,
                    },
                },
            },
        )
    )
    assert isinstance(scores["weighted"], CompositeScore)
    assert scores["weighted"]("10", "10")["value"] == pytest.approx(1.0)


def test_domain_specific_score_from_configs_with_nested_composite_raw_dicts() -> None:
    scores = DomainSpecificScore.from_configs(
        outer=ScoreFnConfig(
            type="composite",
            kwargs={
                "aggregation": "sub_scores.score.inner",
                "sub_scores": {
                    "inner": {
                        "type": "composite",
                        "aggregation": "sub_scores.score.leaf",
                        "sub_scores": {
                            "leaf": {
                                "type": "log_scaled_score",
                                "tolerance": 0.1,
                                "scaling": 10.0,
                            },
                        },
                    },
                },
            },
        )
    )
    result = scores["outer"]("10", "10")
    assert result["value"] == pytest.approx(1.0)


def test_cvss_score_get_cvss_score() -> None:
    # Test valid CVSS vector
    cvss_vector = "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    score_fn = CVSSScore()
    score = score_fn.get_cvss_score(cvss_vector)
    assert 0 <= score <= 10, "CVSS score should be normalized to [0, 10]"

    # Test invalid CVSS vector
    with pytest.raises(CVSS3MalformedError, match="Malformed CVSS3 vector"):
        score_fn.get_cvss_score("INVALID:VECTOR")


def test_cvss_score() -> None:
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


def test_cvss_score_aggregate() -> None:
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


def test_alias_accuracy_score() -> None:
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


def test_alias_accuracy_score_aggregate() -> None:
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


def test_composite_score_from_configs() -> None:
    scores = DomainSpecificScore.from_configs(
        weighted_score=ScoreFnConfig(
            type="composite",
            kwargs={
                "aggregation": "sub_scores.weight.cvss_score * sub_scores.score.cvss_score + sub_scores.weight.jaccard_index * sub_scores.score.jaccard_index",
                "sub_scores": {
                    "cvss_score": ScoreFnConfig(
                        type="cvss", kwargs={"attributes": {"weight": 0.7}}
                    ),
                    "jaccard_index": ScoreFnConfig(
                        type="jaccard", kwargs={"attributes": {"weight": 0.3}}
                    ),
                },
            },
        )
    )
    assert set(scores.keys()) == {"weighted_score"}
    assert isinstance(scores["weighted_score"], CompositeScore)

    with pytest.raises(
        ValueError, match="Invalid aggregation expression for composite score"
    ):
        DomainSpecificScore.from_configs(
            invalid_composite=ScoreFnConfig(
                type="composite",
                kwargs={
                    "aggregation": "sum(sub_scores.score[k] * sub_scores.weight[k] for k in sub_scores.score.keys())",
                    "sub_scores": {
                        "cvss_score": ScoreFnConfig(type="cvss"),
                        "jaccard_index": ScoreFnConfig(type="jaccard"),
                    },
                },
            )
        )


def test_composite_score() -> None:
    score_fn = CompositeScore(
        aggregation="0.7 * sub_scores.score.log_1 + 0.3 * sub_scores.score.log_2",
        sub_scores={
            "log_1": ScoreFnConfig(
                type="log_scaled_score", kwargs={"tolerance": 0.1, "scaling": 10.0}
            ),
            "log_2": ScoreFnConfig(
                type="log_scaled_score", kwargs={"tolerance": 0.5, "scaling": 2.0}
            ),
        },
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
        aggregation="sum(sub_scores.score[k] * sub_scores.weight[k] for k in sub_scores.score.keys()) / sum(sub_scores.weight.values())",
        sub_scores={
            "log_1": ScoreFnConfig(
                type="log_scaled_score",
                kwargs={"tolerance": 0.1, "scaling": 10.0, "attributes": {"weight": 3}},
            ),
            "log_2": ScoreFnConfig(
                type="log_scaled_score",
                kwargs={"tolerance": 0.5, "scaling": 2.0, "attributes": {"weight": 1}},
            ),
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
        aggregation="0.7 * sub_scores.score.log_1 + 0.3 * sub_scores.score.log_2",
        sub_scores={
            "log_1": ScoreFnConfig(
                type="log_scaled_score", kwargs={"tolerance": 0.1, "scaling": 10.0}
            ),
            "log_2": ScoreFnConfig(
                type="log_scaled_score", kwargs={"tolerance": 0.5, "scaling": 2.0}
            ),
        },
    )

    scores = [
        Score(value=0.2, raw_value=0.2),
        Score(value=0.6, raw_value=0.6),
        Score(value=0.8, raw_value=0.8),
    ]
    aggregated_scores = score_fn.aggregate(scores)
    assert aggregated_scores == {"mean": pytest.approx(8 / 15)}
