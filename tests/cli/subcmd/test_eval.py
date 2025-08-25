# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest

from faith.cli.subcmd.eval import RecordHandlingParams, compute_experiment_metrics


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compute_experiment_metrics() -> None:
    with patch("faith.cli.subcmd.eval.write_as_json", return_value=None):
        assert compute_experiment_metrics(
            experiment_path=Path(__file__).parent
            / "testdata"
            / "eval"
            / "bench_1"
            / "experiment.json",
            record_params=RecordHandlingParams(
                annotate_prediction_stats=False,
                recompute_stats=False,
            ),
        ) == {
            "stats": {
                "accuracy": {
                    "max": pytest.approx(3 / 4),
                    "mean": pytest.approx(41 / 56),
                    "median": pytest.approx(41 / 56),
                    "min": pytest.approx(5 / 7),
                    "p_25": pytest.approx(81 / 112),
                    "p_75": pytest.approx(83 / 112),
                    "std": pytest.approx(1 / 56),
                },
                "confusion_matrix": {
                    "mean_per_query": {
                        "": {
                            "": 0,
                            "A": 0,
                            "B": 0,
                            "C": 0,
                            "D": 0,
                        },
                        "A": {
                            "": pytest.approx(1 / 11),
                            "A": pytest.approx(3 / 11),
                            "B": 0,
                            "C": pytest.approx(2 / 11),
                            "D": 0,
                        },
                        "B": {
                            "": 0,
                            "A": 0,
                            "B": pytest.approx(1 / 11),
                            "C": 0,
                            "D": 0,
                        },
                        "C": {
                            "": 0,
                            "A": 0,
                            "B": 0,
                            "C": pytest.approx(4 / 11),
                            "D": 0,
                        },
                        "D": {
                            "": 0,
                            "A": 0,
                            "B": 0,
                            "C": 0,
                            "D": 0,
                        },
                    },
                    "mean_per_trial": {
                        "": {
                            "": 0,
                            "A": 0,
                            "B": 0,
                            "C": 0,
                            "D": 0,
                        },
                        "A": {
                            "": pytest.approx(1 / 2),
                            "A": pytest.approx(3 / 2),
                            "B": 0,
                            "C": 1,
                            "D": 0,
                        },
                        "B": {
                            "": 0,
                            "A": 0,
                            "B": pytest.approx(1 / 2),
                            "C": 0,
                            "D": 0,
                        },
                        "C": {
                            "": 0,
                            "A": 0,
                            "B": 0,
                            "C": 2,
                            "D": 0,
                        },
                        "D": {
                            "": 0,
                            "A": 0,
                            "B": 0,
                            "C": 0,
                            "D": 0,
                        },
                    },
                    "total": {
                        "": {
                            "": 0,
                            "A": 0,
                            "B": 0,
                            "C": 0,
                            "D": 0,
                        },
                        "A": {
                            "": 1,
                            "A": 3,
                            "B": 0,
                            "C": 2,
                            "D": 0,
                        },
                        "B": {
                            "": 0,
                            "A": 0,
                            "B": 1,
                            "C": 0,
                            "D": 0,
                        },
                        "C": {
                            "": 0,
                            "A": 0,
                            "B": 0,
                            "C": 4,
                            "D": 0,
                        },
                        "D": {
                            "": 0,
                            "A": 0,
                            "B": 0,
                            "C": 0,
                            "D": 0,
                        },
                    },
                },
                "f1_scores": {
                    "A": {
                        "max": pytest.approx(2 / 3),
                        "mean": pytest.approx(2 / 3),
                        "median": pytest.approx(2 / 3),
                        "min": pytest.approx(2 / 3),
                        "p_25": pytest.approx(2 / 3),
                        "p_75": pytest.approx(2 / 3),
                        "std": 0,
                    },
                    "B": {
                        "max": pytest.approx(float("nan"), nan_ok=True),
                        "mean": pytest.approx(float("nan"), nan_ok=True),
                        "median": pytest.approx(float("nan"), nan_ok=True),
                        "min": pytest.approx(float("nan"), nan_ok=True),
                        "p_25": pytest.approx(float("nan"), nan_ok=True),
                        "p_75": pytest.approx(float("nan"), nan_ok=True),
                        "std": pytest.approx(float("nan"), nan_ok=True),
                    },
                    "C": {
                        "max": 1,
                        "mean": pytest.approx(5 / 6),
                        "median": pytest.approx(5 / 6),
                        "min": pytest.approx(2 / 3),
                        "p_25": pytest.approx(3 / 4),
                        "p_75": pytest.approx(11 / 12),
                        "std": pytest.approx(1 / 6),
                    },
                    "D": {
                        "max": pytest.approx(float("nan"), nan_ok=True),
                        "mean": pytest.approx(float("nan"), nan_ok=True),
                        "median": pytest.approx(float("nan"), nan_ok=True),
                        "min": pytest.approx(float("nan"), nan_ok=True),
                        "p_25": pytest.approx(float("nan"), nan_ok=True),
                        "p_75": pytest.approx(float("nan"), nan_ok=True),
                        "std": pytest.approx(float("nan"), nan_ok=True),
                    },
                },
                "format": {
                    "mean_per_query": {
                        "improper": 0,
                        "inferred": 0,
                        "invalid": pytest.approx(1 / 11),
                        "proper": pytest.approx(10 / 11),
                    },
                    "mean_per_trial": {
                        "improper": 0,
                        "inferred": 0,
                        "invalid": pytest.approx(1 / 2),
                        "proper": 5,
                    },
                    "total": {
                        "improper": 0,
                        "inferred": 0,
                        "invalid": 1,
                        "proper": 10,
                    },
                },
                "format_breakdown": {
                    "mean_per_trial": {
                        "improper": {
                            "correct": pytest.approx(0),
                            "incorrect": pytest.approx(0),
                        },
                        "inferred": {
                            "correct": pytest.approx(0),
                            "incorrect": pytest.approx(0),
                        },
                        "invalid": {
                            "correct": pytest.approx(0),
                            "incorrect": pytest.approx(1 / 2),
                        },
                        "proper": {
                            "correct": pytest.approx(4),
                            "incorrect": pytest.approx(1),
                        },
                    },
                    "mean_per_query": {
                        "improper": {
                            "correct": pytest.approx(0),
                            "incorrect": pytest.approx(0),
                        },
                        "inferred": {
                            "correct": pytest.approx(0),
                            "incorrect": pytest.approx(0),
                        },
                        "invalid": {
                            "correct": pytest.approx(0),
                            "incorrect": pytest.approx(1 / 11),
                        },
                        "proper": {
                            "correct": pytest.approx(8 / 11),
                            "incorrect": pytest.approx(2 / 11),
                        },
                    },
                    "total": {
                        "improper": {"correct": 0, "incorrect": 0},
                        "inferred": {"correct": 0, "incorrect": 0},
                        "invalid": {"correct": 0, "incorrect": 1},
                        "proper": {"correct": 8, "incorrect": 2},
                    },
                },
                "format_rate": {
                    "improper": {
                        "max": 0,
                        "mean": 0,
                        "median": 0,
                        "min": 0,
                        "p_25": 0,
                        "p_75": 0,
                        "std": 0,
                    },
                    "inferred": {
                        "max": 0,
                        "mean": 0,
                        "median": 0,
                        "min": 0,
                        "p_25": 0,
                        "p_75": 0,
                        "std": 0,
                    },
                    "invalid": {
                        "max": pytest.approx(1 / 4),
                        "mean": pytest.approx(1 / 8),
                        "median": pytest.approx(1 / 8),
                        "min": 0,
                        "p_25": pytest.approx(1 / 16),
                        "p_75": pytest.approx(3 / 16),
                        "std": pytest.approx(1 / 8),
                    },
                    "proper": {
                        "max": 1,
                        "mean": pytest.approx(7 / 8),
                        "median": pytest.approx(7 / 8),
                        "min": pytest.approx(3 / 4),
                        "p_25": pytest.approx(13 / 16),
                        "p_75": pytest.approx(15 / 16),
                        "std": pytest.approx(1 / 8),
                    },
                },
                "label": {
                    "mean_per_query": {
                        "A": pytest.approx(6 / 11),
                        "B": pytest.approx(1 / 11),
                        "C": pytest.approx(4 / 11),
                        "D": pytest.approx(0),
                    },
                    "mean_per_trial": {
                        "A": pytest.approx(3),
                        "B": pytest.approx(1 / 2),
                        "C": pytest.approx(2),
                        "D": pytest.approx(0),
                    },
                    "total": {
                        "A": pytest.approx(6),
                        "B": pytest.approx(1),
                        "C": pytest.approx(4),
                        "D": pytest.approx(0),
                    },
                },
                "lenient_accuracy": {
                    "max": pytest.approx(3 / 4),
                    "mean": pytest.approx(41 / 56),
                    "median": pytest.approx(41 / 56),
                    "min": pytest.approx(5 / 7),
                    "p_25": pytest.approx(81 / 112),
                    "p_75": pytest.approx(83 / 112),
                    "std": pytest.approx(1 / 56),
                },
                "mean_output_tokens": {
                    "max": pytest.approx(101 / 4),
                    "mean": pytest.approx(197 / 8),
                    "median": pytest.approx(197 / 8),
                    "min": pytest.approx(24),
                    "p_25": pytest.approx(389 / 16),
                    "p_75": pytest.approx(399 / 16),
                    "std": pytest.approx(5 / 8),
                },
                "num_trials": 2,
                "rate_max_token_halt": {
                    "max": pytest.approx(2 / 7),
                    "mean": pytest.approx(15 / 56),
                    "median": pytest.approx(15 / 56),
                    "min": pytest.approx(1 / 4),
                    "p_25": pytest.approx(29 / 112),
                    "p_75": pytest.approx(31 / 112),
                    "std": pytest.approx(1 / 56),
                },
                "queries": {"mean": pytest.approx(11 / 2), "total": 11},
                "weighted_avg_f1": {
                    "max": pytest.approx(5 / 6),
                    "mean": pytest.approx(65 / 84),
                    "median": pytest.approx(65 / 84),
                    "min": pytest.approx(5 / 7),
                    "p_25": pytest.approx(125 / 168),
                    "p_75": pytest.approx(45 / 56),
                    "std": pytest.approx(0.05952380952380948),
                },
            },
            "per_trial_metrics": {
                "trials/373363/e84f2e324500232d": {
                    "accuracy": pytest.approx(3 / 4),
                    "confusion_matrix_count": {
                        "A": {"A": 1, "B": 0, "C": 0, "D": 0, "": 1},
                        "B": {"A": 0, "B": 0, "C": 0, "D": 0, "": 0},
                        "C": {"A": 0, "B": 0, "C": 2, "D": 0, "": 0},
                        "D": {"A": 0, "B": 0, "C": 0, "D": 0, "": 0},
                        "": {"A": 0, "B": 0, "C": 0, "D": 0, "": 0},
                    },
                    "f1_scores": {
                        "A": pytest.approx(2 / 3),
                        "B": pytest.approx(float("nan"), nan_ok=True),
                        "C": pytest.approx(1),
                        "D": pytest.approx(float("nan"), nan_ok=True),
                    },
                    "format_breakdown_count": {
                        "improper": {"correct": 0, "incorrect": 0},
                        "inferred": {"correct": 0, "incorrect": 0},
                        "invalid": {"correct": 0, "incorrect": 1},
                        "proper": {"correct": 3, "incorrect": 0},
                    },
                    "format_count": {
                        "improper": 0,
                        "inferred": 0,
                        "invalid": 1,
                        "proper": 3,
                    },
                    "format_rate": {
                        "improper": pytest.approx(0),
                        "inferred": pytest.approx(0),
                        "invalid": pytest.approx(1 / 4),
                        "proper": pytest.approx(3 / 4),
                    },
                    "label_count": {"A": 2, "B": 0, "C": 2, "D": 0},
                    "lenient_accuracy": pytest.approx(0.75),
                    "mean_output_tokens": pytest.approx(101 / 4),
                    "query_count": 4,
                    "rate_max_token_halt": pytest.approx(1 / 4),
                    "weighted_avg_f1": pytest.approx(5 / 6),
                },
                "trials/373365/099186c0f688a67a": {
                    "accuracy": pytest.approx(5 / 7),
                    "confusion_matrix_count": {
                        "A": {"A": 2, "B": 0, "C": 2, "D": 0, "": 0},
                        "B": {"A": 0, "B": 1, "C": 0, "D": 0, "": 0},
                        "C": {"A": 0, "B": 0, "C": 2, "D": 0, "": 0},
                        "D": {"A": 0, "B": 0, "C": 0, "D": 0, "": 0},
                        "": {"A": 0, "B": 0, "C": 0, "D": 0, "": 0},
                    },
                    "f1_scores": {
                        "A": pytest.approx(2 / 3),
                        "B": pytest.approx(1),
                        "C": pytest.approx(2 / 3),
                        "D": pytest.approx(float("nan"), nan_ok=True),
                    },
                    "format_breakdown_count": {
                        "improper": {"correct": 0, "incorrect": 0},
                        "inferred": {"correct": 0, "incorrect": 0},
                        "invalid": {"correct": 0, "incorrect": 0},
                        "proper": {"correct": 5, "incorrect": 2},
                    },
                    "format_count": {
                        "improper": 0,
                        "inferred": 0,
                        "invalid": 0,
                        "proper": 7,
                    },
                    "format_rate": {
                        "improper": pytest.approx(0),
                        "inferred": pytest.approx(0),
                        "invalid": pytest.approx(0),
                        "proper": pytest.approx(1),
                    },
                    "label_count": {"A": 4, "B": 1, "C": 2, "D": 0},
                    "lenient_accuracy": pytest.approx(5 / 7),
                    "mean_output_tokens": pytest.approx(24),
                    "query_count": 7,
                    "rate_max_token_halt": pytest.approx(2 / 7),
                    "weighted_avg_f1": pytest.approx(5 / 7),
                },
            },
        }
