#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Execute the authoring page's examples without a model service or containers.

Only model creation, environment construction, and GPU cleanup are replaced.
The real CLI, tool loop, registries, checkpointing, grading, and CSV summarizer
run against temporary fixtures extracted from the documentation. The example
registrations are installed in memory, never in the source tree.
"""

from __future__ import annotations

import contextlib
import csv
import json
import re
import shlex
import shutil
import sys
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from faith._internal.io.json import read_json_file
from faith._types.config.patterns import AnswerFormat
from faith._types.model.prompt import PromptFormatter
from faith.agentic import orchestration, registries
from faith.agentic.core import AnswerExtractor, Environment, Task, TaskLoader
from faith.agentic.core.environment import EnvironmentUnavailableError
from faith.cli.subcmd import query as query_module
from faith.cli.subcmd.summarize import _process_metrics_file
from faith.experiment import trial_runner
from faith.record_pipelines.agentic_formatting import TaskFormatter
from faith.record_pipelines.trial_log import read_trial_log

PAGE = (
    Path(__file__).resolve().parents[1]
    / "docs/benchmarks/creating-agentic-benchmarks.md"
)

# Executing the repository-owned snippets is the purpose of this regression test.
# pylint: disable=exec-used


def snippet(title: str) -> str:
    """Read one uniquely titled fenced block from the live documentation."""
    blocks = re.findall(
        rf'^```\w+ title="{re.escape(title)}"\n(.*?)^```',
        PAGE.read_text(),
        re.M | re.S,
    )
    if len(blocks) != 1:
        raise AssertionError(f"Expected one example block titled {title!r}")
    return blocks[0]


def completion(tool: str | None = None, arguments: dict | None = None):
    """An API-shaped response, without requiring the optional OpenAI SDK."""
    calls = []
    if tool is not None:
        calls.append(
            SimpleNamespace(
                id=f"call-{tool}",
                function=SimpleNamespace(name=tool, arguments=json.dumps(arguments)),
            )
        )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=calls),
                finish_reason="tool_calls" if calls else "stop",
            )
        ]
    )


class MemoryEnvironment(Environment):
    """A lifecycle/command double; it does not test container isolation."""

    def __init__(self, config, mode: str):
        self.config = config
        self.mode = mode
        self.files: dict[str, bytes] = {}
        self.sealed = False
        self.setups = 0
        self.teardowns = 0

    def setup(self, task: Task) -> None:
        """Load the task payload into memory and apply its protection policy."""
        self.setups += 1
        if self.mode == "environment_error":
            raise EnvironmentUnavailableError("scripted setup failure")
        if task.data_provider is not None:
            task.data_provider()
        assert task.data_path is not None and task.data_path.is_file()
        self.files = {task.data_path.name: task.data_path.read_bytes()}
        self.sealed = False
        if self.config.read_only_workspace:
            self.seal_workspace()

    def seal_workspace(self) -> None:
        """Record workspace protection without modifying real permissions."""
        self.sealed = True

    def run_command(self, command: str) -> tuple[int, str]:
        """Simulate only the manifest-reading command used by this example."""
        assert self.sealed, "The example requires protection before commands"
        assert command == "cat manifest.txt", "Never execute model-supplied shell"
        return 0, self.files["manifest.txt"].decode("utf-8")

    def teardown(self) -> None:
        """Record that the test environment was released."""
        self.teardowns += 1


class AuthoringExampleTests(unittest.TestCase):
    """Keep the published example executable and its guarantees explicit."""

    # Keep independently reported cases with their shared pipeline fixture.
    # pylint: disable=too-many-public-methods

    parser: ArgumentParser

    @classmethod
    def setUpClass(cls):
        # cli.py parses argv during import; unittest's argv is not a FAITH command.
        with patch.object(
            sys, "argv", ["faith", "summarize", "--experiment-path", "/tmp"]
        ):
            # pylint: disable-next=import-outside-toplevel
            from faith.cli.cli import _cli_parser

        cls.parser = _cli_parser

    def setUp(self):
        # unittest retains this context through the complete test lifecycle.
        # pylint: disable=consider-using-with
        self.root = Path(
            self.enterContext(tempfile.TemporaryDirectory(prefix="faith-doc-example-"))
        )
        self.enterContext(contextlib.chdir(self.root))
        # Fail closed if a missing mock could reach the network, a daemon, or a
        # process-management command. These are guards, not simulated successes.
        for target in ("socket.create_connection", "subprocess.Popen", "os.system"):
            self.enterContext(
                patch(target, side_effect=AssertionError(f"Unexpected {target}"))
            )

        self._register_example()
        self.benchmark = self.root / "custom-file-list"
        task_dir = self.benchmark / "tasks/case-001"
        task_dir.mkdir(parents=True)
        (self.benchmark / "benchmark.yaml").write_text(
            snippet("custom-file-list/benchmark.yaml")
        )
        for name in ("manifest.txt", "expected.json"):
            (task_dir / name).write_text(snippet(name))

    def _register_example(self):
        module_name = "faith.agentic.file_list_example"
        module = ModuleType(module_name)
        exec(
            compile(snippet("file_list_example.py"), str(PAGE), "exec"), module.__dict__
        )
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module
        if previous is None:
            self.addCleanup(sys.modules.pop, module_name, None)
        else:
            self.addCleanup(sys.modules.__setitem__, module_name, previous)
        self.loader_class: type[TaskLoader] = module.__dict__["FileListTaskLoader"]
        self.extractor_class: type[AnswerExtractor] = module.__dict__[
            "FileListAnswerExtractor"
        ]
        imported: dict = {}
        exec(snippet("Example registry import"), imported)
        members = snippet("Example registry members")
        for name in ("AgenticTaskLoaderType", "AgenticAnswerExtractorType"):
            section = members.split(f"# Inside {name}:\n", 1)[1].split("\n\n", 1)[0]
            namespace = imported.copy()
            exec(section, namespace)
            original = getattr(registries, name)
            entries = {member.name: member.value for member in original}
            entries["FILE_LIST_EXAMPLE"] = namespace["FILE_LIST_EXAMPLE"]
            extended = original.__bases__[0](name, entries)
            self.enterContext(patch.object(registries, name, extended))
            self.enterContext(patch.object(orchestration, name, extended))

    def command(self, title: str, *overrides: str) -> None:
        """Execute one documented CLI command with optional test overrides."""
        tokens = shlex.split(snippet(title).replace("\\\n", ""))
        self.assertEqual(tokens[0], "faith")
        args = self.parser.parse_args(tokens[1:] + list(overrides))
        args.func(args)

    def pipeline(self, mode: str, output: str = "out/file-list"):
        """Run all three documented CLI commands, substituting only boundaries."""
        responses = 0
        environments = []

        def respond(**kwargs):
            nonlocal responses
            if kwargs["tool_choice"] == "required":
                return completion("probe", {"x": "ok"})
            responses += 1
            if mode == "error":
                raise RuntimeError("scripted API failure")
            if mode == "missing":
                return completion()
            if responses == 1:
                return completion("bash_exec", {"command": "cat manifest.txt"})
            answer = "not JSON" if mode == "malformed" else '["src/parser.py"]'
            return completion("submit", {"answer": answer})

        def environment(config):
            instance = MemoryEnvironment(config, mode)
            environments.append(instance)
            return instance

        model = SimpleNamespace(
            client=SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=respond))
            ),
            name_or_path="scripted-model",
            supported_formats=[PromptFormatter.CHAT],
            api_max_attempts=1,
            api_retry_sleep_secs=0,
        )
        csv_path = (
            self.root
            / f"{output}-summary-{mode}-{len(list(self.root.rglob('*.csv')))}.csv"
        )
        with (
            patch.object(query_module, "create_model", return_value=model),
            patch.object(
                trial_runner,
                "build_environment",
                side_effect=environment,
            ),
            patch.object(query_module, "_killall_gpu_processes") as cleanup,
        ):
            self.command("Query the custom example", "--datastore-location", output)
            cleanup.assert_called_once()
        self.command("Evaluate the custom example", "--experiment-path", output)
        self.command(
            "Summarize the custom example",
            "--experiment-path",
            output,
            "--summary-filepath",
            str(csv_path),
        )
        metrics, record, metric_path, row = self._results(csv_path, output)
        return metrics, record, metric_path, responses, environments, row

    def _results(self, csv_path: Path, output: str):
        metric_paths = list((self.root / output).rglob("metrics.json"))
        self.assertEqual(len(metric_paths), 1)
        with csv_path.open() as csv_file:
            rows = list(csv.DictReader(csv_file))
        self.assertEqual(len(rows), 1)
        self.assertIn("error_rate_mean", rows[0])
        self.assertIn("stopped_reason_total", rows[0])
        log_paths = list((self.root / output).rglob("benchmark-log.json"))
        self.assertEqual(len(log_paths), 1)
        records = list(read_trial_log(log_paths[0]))
        self.assertEqual(len(records), 1)
        return (
            read_json_file(metric_paths[0]),
            records[0],
            metric_paths[0],
            rows[0],
        )

    def test_preflight_and_registrations(self):
        exec(
            compile(snippet("Validate the example before querying"), str(PAGE), "exec"),
            {},
        )
        self.assertIs(
            registries.AgenticTaskLoaderType("file_list_example").impl_cls,
            self.loader_class,
        )
        self.assertIn(
            "VULNERABILITY_LOCALIZATION", registries.AgenticTaskLoaderType.__members__
        )

    def test_correct_submission_through_query_eval_and_csv(self):
        metrics, record, path, calls, envs, row = self.pipeline("correct")
        self.assertEqual(calls, 2)
        self.assertEqual(metrics["stats"]["mean_file_f1"]["mean"], 1.0)
        self.assertEqual(float(row["mean_file_f1_mean"]), 1.0)
        self.assertEqual(metrics["stats"]["stopped_reason"]["total"]["submitted"], 1)
        primary = [item for item in _process_metrics_file(path) if item["is_primary"]]
        self.assertEqual(
            [item["metric_name"] for item in primary], ["mean_file_f1.mean"]
        )
        self.assertEqual(record.stats.prediction, ["src/parser.py"])
        self.assertEqual(envs[0].setups, envs[0].teardowns)
        self.assertEqual(set(envs[0].files), {"manifest.txt"})

    def test_malformed_submission_scores_zero(self):
        metrics, record, *_ = self.pipeline("malformed")
        self.assertEqual(metrics["stats"]["mean_file_f1"]["mean"], 0.0)
        self.assertEqual(record.stats.answer_format, AnswerFormat.IMPROPER)
        self.assertEqual(metrics["stats"]["error_rate"]["mean"], 0.0)

    def test_missing_submission_scores_zero(self):
        metrics, record, *_ = self.pipeline("missing")
        self.assertEqual(metrics["stats"]["mean_file_f1"]["mean"], 0.0)
        self.assertEqual(record.stats.answer_format, AnswerFormat.INVALID)
        self.assertEqual(
            metrics["stats"]["stopped_reason"]["total"]["no_tool_calls"], 1
        )

    def test_failures_are_excluded_and_reported(self):
        for mode in ("error", "environment_error"):
            with self.subTest(mode=mode):
                metrics, record, *_ = self.pipeline(mode, f"out/{mode}")
                self.assertNotIn("mean_file_f1", metrics["stats"])
                self.assertEqual(metrics["stats"]["error_rate"]["mean"], 1.0)
                self.assertEqual(metrics["stats"]["stopped_reason"]["total"][mode], 1)
                trial = next(iter(metrics["per_trial_metrics"].values()))
                self.assertEqual(trial["query_count"], 1)
                self.assertEqual(trial["scored_count"], 0)
                self.assertIsNotNone(record.stats.error)

    def test_resume_reuses_trajectory_but_payload_change_invalidates_it(self):
        self.pipeline("correct")
        _, _, _, calls, environments, _ = self.pipeline("correct")
        self.assertEqual(calls, 0)
        self.assertTrue(all(env.setups == 0 for env in environments))
        payload = self.benchmark / "tasks/case-001/manifest.txt"
        payload.write_text(payload.read_text() + "src/extra.py other\n")
        _, _, _, calls, _, _ = self.pipeline("correct")
        self.assertEqual(calls, 2)

    def test_identity_survives_moving_the_payload_cache(self):
        moved = self.root / "another-cache"
        shutil.copytree(self.benchmark, moved)
        loader = self.loader_class({}, "/workspace")
        tasks = [loader.load_tasks(path)[0] for path in (self.benchmark, moved)]
        records = list(enumerate(tasks[:1]) >> TaskFormatter("docs", None))
        moved_records = list(enumerate(tasks[1:]) >> TaskFormatter("docs", None))
        self.assertNotEqual(tasks[0].data_path, tasks[1].data_path)
        self.assertEqual(
            records[0].data.benchmark_sample_hash,
            moved_records[0].data.benchmark_sample_hash,
        )

    def test_extractor_rejects_wrong_shapes_and_terminal_tools(self):
        extractor = self.extractor_class({})
        invalid_answers: list[tuple[str | None, str | list | None]] = [
            (None, None),
            ("other", "[]"),
            ("submit", []),
            ("submit", "{}"),
            ("submit", "[1]"),
        ]
        for tool, answer in invalid_answers:
            with self.subTest(tool=tool, answer=answer):
                self.assertIsNone(extractor.extract(tool, answer))
        self.assertEqual(extractor.extract("submit", "[]"), [])

    def test_loader_sorts_tasks_and_uses_the_sandbox_working_directory(self):
        for name in ("case-003", "case-002"):
            shutil.copytree(
                self.benchmark / "tasks/case-001", self.benchmark / "tasks" / name
            )
        tasks = self.loader_class({}, "/workspace/repo").load_tasks(self.benchmark)
        self.assertEqual(
            [task.id for task in tasks], ["case-001", "case-002", "case-003"]
        )
        self.assertTrue(all("/workspace/repo" in task.prompt for task in tasks))
        self.assertTrue(all(str(self.root) not in task.prompt for task in tasks))


if __name__ == "__main__":
    unittest.main()
