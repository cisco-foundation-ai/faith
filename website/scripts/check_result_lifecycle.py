#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the saved-result limitations documented by the site, without services.

These are documentation contract checks, not assertions of desired runtime
behavior. When an implementation limitation is fixed, update its documentation
and the corresponding check together. Evaluation and export use real FAITH
code; the BigQuery insertion method uses a mock client, and the view SELECTs
run in SQLite (not a BigQuery integration test). No engine extras are required.
"""

from __future__ import annotations

import ast
import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from faith._internal.algo.hash import dict_sha256
from faith._internal.io.json import write_as_json
from faith._types.config.patterns import AnswerFormat, CaptureTransform, PatternDef
from faith._types.record.model import ModelRecord
from faith._types.record.model_response import ChatResponse
from faith._types.record.prompt import PromptRecord
from faith._types.record.sample import Metadata, ReplacementStrategy, SampleRecord
from faith.benchmark.scores.domain_specific import LLMJudgeScore
from faith.cli.subcmd.eval import _agg_trials, evaluate_experiment
from faith.cli.subcmd.summarize import _process_metrics_file
from faith.experiment.summarize import build_summary
from faith.record_pipelines.params import RecordHandlingParams

REPO = Path(__file__).resolve().parents[2]


class ResultLifecycleTests(unittest.TestCase):
    """Pin saved-result behavior using temporary logs and mocked services."""

    def setUp(self):
        # unittest closes this context after each case, even on failure.
        # pylint: disable-next=consider-using-with
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(
            patch("socket.create_connection", side_effect=AssertionError("network"))
        )
        self.enterContext(
            patch("subprocess.Popen", side_effect=AssertionError("subprocess"))
        )
        self.experiment_path = self.root / "experiment.json"
        self.metrics_path = self.root / "metrics.json"
        self.log_path = self.root / "trial/benchmark-log.json"
        self.experiment: dict[str, Any] = {
            "metadata": {"version": "original-run-version"},
            "benchmark_config": {
                "mcqa_config": {"answer_symbols": ["A", "B"]},
                "output_processing": {
                    "primary_metric": "accuracy.mean",
                    "answer_formats": [
                        {"format_type": "proper", "pattern": "Answer: ([AB])"}
                    ],
                },
            },
            "experiment_params": {
                "benchmark": {
                    "name": "saved-result-fixture",
                    "generation_mode": "chat_comp",
                    "prompt_format": "chat",
                    "n_shot": "0",
                },
                "model": {
                    "name": "fixture-model",
                    "path": "fixture-model",
                    "generation": {"temperature": 0.0},
                },
            },
            "trial_records": {"trial": {"trial_log_path": "trial/benchmark-log.json"}},
        }
        record = SampleRecord(
            metadata=Metadata(version="original-run-version"),
            data=PromptRecord(
                benchmark_sample_index=0,
                benchmark_sample_hash="fixture",
                subject=None,
                system_prompt=None,
                instruction=None,
                question="Select A.",
                choices={"A": "correct", "B": "incorrect"},
                label="A",
                formatted_question="Select A.",
                formatted_answer="Answer: A",
                question_prompt="Select A.",
            ),
            model_data=ModelRecord(
                prompt="Select A.",
                answer_symbol_ids={},
                chat_comp=ChatResponse(
                    output_text="Answer: A", answer_text="Answer: A"
                ),
            ),
        )
        record.metadata.data_hash = dict_sha256(record.data.to_dict())
        record.metadata.model_data_hash = dict_sha256(record.model_data.to_dict())
        self.record = record.to_dict()
        write_as_json(self.experiment_path, self.experiment)
        write_as_json(self.log_path, [self.record])

    def evaluate(self, strategy=ReplacementStrategy.IF_HASH_DIFFERS):
        """Grade the saved fixture with the requested replacement policy."""
        return evaluate_experiment(
            self.experiment_path,
            RecordHandlingParams(replacement_strategy=strategy),
            metrics_output_path=self.metrics_path,
        )

    def test_regrading_uses_snapshot_and_requires_explicit_invalidation(self):
        self.assertEqual(self.evaluate()["stats"]["accuracy"]["mean"], 1)
        changed = copy.deepcopy(self.experiment["benchmark_config"])
        changed["output_processing"]["answer_formats"][0]["pattern"] = "Verdict: ([AB])"
        # JSON is valid YAML: changing a neighboring source config has no effect.
        write_as_json(self.root / "benchmark.yaml", {"benchmark": changed})
        self.assertEqual(
            self.evaluate(ReplacementStrategy.ALWAYS)["stats"]["accuracy"]["mean"], 1
        )
        self.experiment["benchmark_config"] = changed
        write_as_json(self.experiment_path, self.experiment)
        self.assertEqual(self.evaluate()["stats"]["accuracy"]["mean"], 1)
        self.assertEqual(
            self.evaluate(ReplacementStrategy.ALWAYS)["stats"]["accuracy"]["mean"], 0
        )

    def test_missing_and_empty_trials_are_omitted(self):
        self.experiment["trial_records"].update(
            missing={"trial_log_path": "missing.json"},
            empty={"trial_log_path": "empty.json"},
        )
        write_as_json(self.root / "empty.json", [])
        write_as_json(self.experiment_path, self.experiment)
        metrics = self.evaluate()
        self.assertEqual(list(metrics["per_trial_metrics"]), ["trial"])
        self.assertEqual(metrics["stats"]["num_trials"], 1)
        self.assertEqual(metrics["stats"]["queries"]["total"], 1)

    def test_summary_does_not_refresh_changed_trial_logs(self):
        self.evaluate()
        write_as_json(self.log_path, [])
        summary = build_summary(self.root, ["accuracy", "queries"])
        self.assertEqual(summary.iloc[0]["accuracy_mean"], 1)
        self.assertEqual(summary.iloc[0]["queries_total"], 1)
        self.assertEqual(self.evaluate()["stats"]["num_trials"], 0)

    def test_trial_means_are_unweighted_and_omit_absent_scores(self):
        trials = {
            "small": {"query_count": 1, "mean_file_f1": 1.0, "error_rate": 0.0},
            "large": {"query_count": 9, "mean_file_f1": 0.0, "error_rate": 0.0},
            "failed": {"query_count": 3, "error_rate": 1.0},
        }
        stats = _agg_trials(trials.values())
        self.assertEqual(stats["num_trials"], 3)
        self.assertEqual(stats["queries"]["total"], 13)
        self.assertEqual(stats["mean_file_f1"]["mean"], 0.5)
        self.assertAlmostEqual(stats["error_rate"]["mean"], 1 / 3)

    def test_judge_calls_its_model_during_scoring(self):
        model = Mock()
        model.query.return_value = [ChatResponse(output_text="1", answer_text="1")]
        with patch(
            "faith.benchmark.scores.domain_specific.create_model", return_value=model
        ) as create:
            scorer = LLMJudgeScore(
                judge_prompt_template="{{ correct_answer }} / {{ generated_answer }}",
                judge_model={"model_engine": "openai", "model_path": "fixture-judge"},
                verdict_formats=[
                    PatternDef(
                        format_type=AnswerFormat.PROPER,
                        pattern=r"(\d)",
                        capture_transform=CaptureTransform(
                            params=["x"], expr="{'awarded_points': float(x)}"
                        ),
                    )
                ],
            )
            self.assertEqual(scorer("A", "A")["value"], 1)
        create.assert_called_once()
        model.query.assert_called_once()

    def test_ingestion_uses_local_path_and_ingester_version(self):
        self.evaluate()
        with patch("faith.ingestion.parser.faith_version", "ingester-version"):
            rows = _process_metrics_file(self.metrics_path)
        self.assertEqual({r["faith_version"] for r in rows}, {"ingester-version"})
        self.assertEqual(
            {r["metrics_file_uri"] for r in rows}, {str(self.metrics_path.absolute())}
        )
        self.assertNotIn("per_trial_metrics", {r["metric_name"] for r in rows})
        self.assertTrue(any(r["is_primary"] for r in rows))

    def test_batch_duplicate_check_inspects_only_first_uri(self):
        # The method and private SDK client are deliberately tested in isolation.
        # pylint: disable=exec-used,protected-access
        # Execute the actual method without importing the optional Google SDK.
        path = REPO / "src/faith/ingestion/storage/bigquery.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if getattr(n, "name", "") == "insert_metrics")
        namespace: dict[str, Any] = {"Any": object}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        client = Mock(table_ref="fixture.metrics")
        client.check_metrics_file_exists.side_effect = lambda uri: uri == "existing"
        client._client.load_table_from_json.return_value.output_rows = 2
        rows = [{"metrics_file_uri": "new"}, {"metrics_file_uri": "existing"}]
        self.assertEqual(namespace["insert_metrics"](client, rows), 2)
        client.check_metrics_file_exists.assert_called_once_with("new")
        client._client.load_table_from_json.assert_called_once_with(
            rows, "fixture.metrics"
        )
        with self.assertRaisesRegex(ValueError, "already ingested"):
            namespace["insert_metrics"](client, list(reversed(rows)))
        self.assertEqual(client._client.load_table_from_json.call_count, 1)


class AnalyticsViewTests(unittest.TestCase):
    """Exercise analytics-view selection semantics with in-memory SQL."""

    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            "CREATE TABLE metrics (metrics_file_uri TEXT, model_key TEXT, benchmark TEXT, "
            "metric_name TEXT, metric_value REAL, is_primary INT, temperature REAL, "
            "top_p REAL, num_shots INT, num_shots_pool_size INT, generation_mode TEXT, "
            "prompt_format TEXT, ingest_time TEXT)"
        )

    def insert(self, name, value, time, temperature=0.0):
        """Insert one metric into the local view fixture."""
        self.db.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "run/metrics.json",
                "model",
                "benchmark",
                name,
                value,
                name == "accuracy.mean",
                temperature,
                1.0,
                0,
                1,
                "chat_comp",
                "chat",
                time,
            ),
        )

    def view(self, name):
        """Evaluate a repository view's SELECT against the local fixture."""
        source = (REPO / f"docs/bigquery/views/{name}.sql").read_text()
        select = source.split(" AS\n", 1)[1].replace(
            "`PROJECT_ID.DATASET_ID.metrics`", "metrics"
        )
        return [dict(row) for row in self.db.execute(select)]

    def test_latest_runs_can_retain_metrics_absent_from_new_ingestion(self):
        self.insert("accuracy.mean", 0.5, "2026-01-01")
        self.insert("queries.total", 100, "2026-01-01")
        self.insert("accuracy.mean", 0.8, "2026-01-02")
        rows = {r["metric_name"]: r for r in self.view("latest_runs")}
        self.assertEqual(rows["accuracy.mean"]["metric_value"], 0.8)
        self.assertEqual(rows["queries.total"]["ingest_time"], "2026-01-01")

    def test_leaderboard_discards_other_configurations_before_filtering(self):
        self.insert("accuracy.mean", 0.5, "2026-01-01", temperature=0.0)
        self.insert("accuracy.mean", 0.8, "2026-01-02", temperature=0.7)
        rows = self.view("leaderboard")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["temperature"], 0.7)
        self.assertEqual([r for r in rows if r["temperature"] == 0.0], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
