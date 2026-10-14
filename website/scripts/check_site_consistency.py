#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Check shared documentation claims and contributor examples without services."""

from __future__ import annotations

import contextlib
import functools
import json
import re
import shlex
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from packaging.specifiers import SpecifierSet

from faith._types.benchmark.sample_ratio import SampleRatio
from faith._types.config.agentic import AgenticEnvironmentConfig
from faith._types.model.generation import GenerationMode, GenParams
from faith._types.model.prompt import PromptFormatter
from faith._types.record.agent import StoppedReason
from faith.experiment.experiment import BenchmarkExperiment

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "website"
FLOW = SITE / "src/components/FlowConcepts/conceptData.js"
LANDING = SITE / "src/components/landing/shared.js"
CONTRIBUTING = SITE / "docs/contributing.md"


def snippet(title: str, page: Path = CONTRIBUTING) -> str:
    """Read one uniquely titled example from a documentation page."""
    blocks = re.findall(
        rf'^```\w+ title="{re.escape(title)}"\n(.*?)^```',
        page.read_text(),
        re.M | re.S,
    )
    if len(blocks) != 1:
        raise AssertionError(f"Expected one example titled {title!r} in {page}")
    return blocks[0]


class SiteConsistencyTests(unittest.TestCase):
    """Check shared presentation claims and runnable documentation examples."""

    # Each public method is an independently reported unittest case.
    # pylint: disable=too-many-public-methods

    def setUp(self):
        self.enterContext(
            patch("socket.create_connection", side_effect=AssertionError("network"))
        )
        self.enterContext(
            patch("subprocess.Popen", side_effect=AssertionError("subprocess"))
        )

    def test_flow_lists_every_runtime_termination_reason(self):
        exits = FLOW.read_text().split("export const EXITS =", 1)[1]
        keys = re.findall(r'key: "([a-z_]+)"', exits)
        labels = re.findall(r'label: "([a-z_]+)"', exits)
        self.assertEqual(keys, labels)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), {str(reason) for reason in StoppedReason})

    def test_flow_does_not_present_runtime_defaults_as_fixed_limits(self):
        component = (SITE / "src/components/FlowConcepts/ForkMerge.js").read_text()
        self.assertIn("configured limits</span>", component)
        self.assertNotRegex(component, r">(?:up to \d+ turns|six ways out)")
        exits = FLOW.read_text().split("export const EXITS =", 1)[1]
        self.assertNotRegex(exits, r"\d+ (?:tool calls|turns)")

    def test_development_examples_select_supported_python_and_existing_extras(self):
        project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
        for title in ("Framework development setup", "Documentation development setup"):
            code = snippet(title)
            version = re.search(r"^uv venv --python (\S+)", code, re.M)
            assert version is not None
            self.assertIn(version.group(1), SpecifierSet(project["requires-python"]))
            self.assertNotIn("SETUPTOOLS_SCM_PRETEND_VERSION", code)
            for extras in re.findall(r"\.\[([a-z,-]+)\]", code):
                self.assertTrue(
                    set(extras.split(",")) <= project["optional-dependencies"].keys()
                )
        self.assertIn(
            "uv pip install -e .\n", snippet("Documentation development setup")
        )
        self.assertNotIn(".[test]", snippet("Documentation development setup"))

    def test_documented_site_scripts_exist_and_match_ci(self):
        package = json.loads((SITE / "package.json").read_text())
        commands = snippet("Documentation development setup").splitlines()
        scripts = [
            shlex.split(command)[2]
            for command in commands
            if command.startswith("npm run ")
        ]
        self.assertTrue({"check-repo", "build", "serve"} <= set(scripts))
        self.assertTrue(set(scripts) <= package["scripts"].keys())
        self.assertIn("npm ci", commands)
        self.assertIn("npm run serve -- --host 127.0.0.1", commands)
        self.assertEqual(
            package["scripts"]["postbuild"], "npm run check-site-navigation"
        )
        workflow = (REPO / ".github/workflows/docs-site.yml").read_text()
        for command in ("npm ci", "npm run check-repo", "npm run build"):
            self.assertIn(command, workflow)

    def test_landing_command_resolves_before_any_model_or_dataset_access(self):
        literal = re.search(
            r"export const CMD_TEXT = `(.*?)`;", LANDING.read_text(), re.S
        )
        assert literal is not None
        command = literal.group(1).replace("\\\\\n", " ")
        argv = shlex.split(command)
        with patch.object(sys, "argv", argv):
            # cli.py parses argv at import time, so import under the fixture.
            # pylint: disable-next=import-outside-toplevel
            from faith.cli.cli import _cli_parser, _cli_query
        args = _cli_parser.parse_args(argv[1:])
        with patch("faith.cli.subcmd.query.run_experiment_queries") as run:
            _cli_query(args, Mock())
        (model,) = run.call_args.args[0]
        self.assertEqual(model.prompt_format, PromptFormatter.CHAT)
        self.assertEqual(args.sample_size, 25)
        self.assertEqual(args.num_trials, 1)
        self.assertEqual(args.stats, ["accuracy"])
        self.assertEqual(str(args.datastore_location), "out/")

    def test_landing_scores_are_labeled_in_rendered_markup(self):
        caption = re.search(
            r"<caption\b[^>]*>(.*?)</caption>", LANDING.read_text(), re.S
        )
        assert caption is not None
        caption_text = " ".join(caption.group(1).split())
        self.assertIn("not measured results", caption_text)
        self.assertIn("Example values", caption_text)
        self.assertIn("output from the command above", caption_text)

    def test_quickstart_backends_match_installation_and_run_the_full_pipeline(self):
        installation = (SITE / "docs/getting-started/installation.md").read_text()
        quickstart = (SITE / "docs/getting-started/quickstart.md").read_text()
        with patch.object(
            sys,
            "argv",
            [
                "faith",
                "query",
                "--model-paths",
                "fixture",
                "--datastore-location",
                "out/fixture",
            ],
        ):
            # pylint: disable-next=import-outside-toplevel
            from faith.cli.cli import _cli_parser, _cli_query
        for engine in ("vllm", "openai"):
            with self.subTest(engine=engine):
                pattern = rf'<TabItem value="{engine}"[^>]*>(.*?)</TabItem>'
                install_panel = re.search(pattern, installation, re.S)
                run_panel = re.search(pattern, quickstart, re.S)
                assert install_panel is not None
                assert run_panel is not None
                self.assertIn(
                    f"uv pip install -e '.[{engine},metrics]'", install_panel[1]
                )
                command = re.search(
                    r"```shell\n(faith run-all\b.*?)```", run_panel[1], re.S
                )
                assert command is not None
                argv = shlex.split(command[1].replace("\\\n", ""))
                args = _cli_parser.parse_args(argv[1:])
                with patch("faith.cli.subcmd.query.run_experiment_queries") as run:
                    _cli_query(args, Mock())
                (model,) = run.call_args.args[0]
                self.assertEqual(str(model.engine.engine_type), engine)
                self.assertEqual(model.prompt_format, PromptFormatter.CHAT)
                self.assertEqual(args.generation_mode, GenerationMode.CHAT_COMP)
                self.assertEqual(args.sample_size, 25)
                self.assertEqual(args.num_trials, 1)
                self.assertEqual(args.benchmarks, ["ctibench-mcqa", "seceval"])
                self.assertEqual(args.stats, ["accuracy"])
                self.assertEqual(str(args.datastore_location), "out/run-001")

    def test_optional_setup_remains_available_after_the_first_run_path(self):
        installation = (SITE / "docs/getting-started/installation.md").read_text()
        quickstart = (SITE / "docs/getting-started/quickstart.md").read_text()
        verify = installation.index("## 3. Verify the installation")
        for section in (
            "### Available extras",
            "### Cloud Storage (optional)",
            "### Packaging for a remote computer",
            "### For development",
            "### Shell autocompletion (optional)",
        ):
            with self.subTest(section=section):
                self.assertGreater(installation.index(section), verify)
        self.assertGreater(
            quickstart.index("### Keeping a long run alive"),
            quickstart.index("## 3. Inspect the results"),
        )
        for target in (
            "../cli/summarize.md#summarize-behavior-flags",
            "../analytics/storage.md#directory-layout",
            "./installation.md#cloud-storage-optional",
            "../configuration/environment-configs.md#cluster-prerequisites",
        ):
            self.assertIn(f"({target})", quickstart)

    def test_local_agentic_serving_steps_share_one_bounded_datastore(self):
        page = SITE / "docs/configuration/serving-agentic-models.md"
        command = shlex.split(
            snippet("Local agentic smoke query", page).replace("\\\n", "")
        )
        with patch.object(sys, "argv", command):
            # pylint: disable-next=import-outside-toplevel
            from faith.cli.cli import _cli_parser, _cli_query
        args = _cli_parser.parse_args(command[1:])
        with patch("faith.cli.subcmd.query.run_experiment_queries") as run:
            _cli_query(args, Mock())
        (model,) = run.call_args.args[0]
        self.assertEqual(str(model.engine.engine_type), "openai")
        self.assertEqual(model.engine.kwargs["base_url"], "http://localhost:8100/v1")
        self.assertEqual(model.prompt_format, PromptFormatter.CHAT)
        self.assertEqual(args.generation_mode, GenerationMode.CHAT_COMP)
        self.assertEqual((args.sample_size, args.num_trials), (1, 1))
        self.assertEqual(args.benchmarks, ["vulnerability-localization-a"])
        commands = snippet("Local agentic smoke results", page).replace("\\\n", "")
        for result_command in commands.splitlines():
            result_args = _cli_parser.parse_args(shlex.split(result_command)[1:])
            self.assertEqual(
                str(result_args.experiment_path), str(args.datastore_location)
            )
        self.assertEqual(
            result_args.stats, ["mean_file_f1", "error_rate", "stopped_reason"]
        )
        self.assertLess(
            page.read_text().index("## 3. Grade and inspect the result"),
            page.read_text().index("## Endpoint diagnostics"),
        )

    def test_qa_walkthrough_loads_its_local_fixture_and_resolves_the_run(self):
        page = SITE / "docs/benchmarks/creating-qa-benchmarks.md"
        # enterContext registers cleanup even if setup or the test fails.
        # pylint: disable-next=consider-using-with
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(contextlib.chdir(root))
        fixture = root / "custom-mcqa"
        fixture.mkdir()
        (fixture / "questions.json").write_text(snippet("questions.json", page))
        (fixture / "benchmark.yaml").write_text(
            snippet("custom-mcqa/benchmark.yaml", page)
        )
        namespace: dict[str, Any] = {}
        # Execute the repository-owned example to detect documentation drift.
        # pylint: disable-next=exec-used
        exec(snippet("Validate the Q&A example", page), namespace)
        config = namespace["config"]
        self.assertEqual(config.metadata.name, "custom-mcqa")
        self.assertEqual(str(config.metadata.state), "enabled")
        self.assertIsNone(config.agentic_config)
        self.assertIsNotNone(config.format.prompt)
        command = shlex.split(snippet("Run the Q&A example", page).replace("\\\n", ""))
        self.assertEqual(command[:2], ["faith", "run-all"])
        with patch.object(sys, "argv", command):
            # pylint: disable-next=import-outside-toplevel
            from faith.cli.cli import _cli_parser, _cli_query
        args = _cli_parser.parse_args(command[1:])
        with patch("faith.cli.subcmd.query.run_experiment_queries") as run:
            _cli_query(args, Mock())
        (model,) = run.call_args.args[0]
        self.assertEqual(str(model.engine.engine_type), "openai")
        self.assertEqual(model.prompt_format, PromptFormatter.CHAT)
        self.assertEqual((args.sample_size, args.num_trials), (1, 1))
        self.assertEqual(args.n_shot, [SampleRatio(0)])
        self.assertEqual([str(p) for p in args.custom_benchmarks], ["custom-mcqa"])
        self.assertEqual(args.stats, ["accuracy", "error_rate"])

    def test_authoring_walkthroughs_link_to_shared_contracts(self):
        reference = (SITE / "docs/benchmarks/creating-benchmarks.md").read_text()
        for target in (
            "./creating-qa-benchmarks.md",
            "./creating-agentic-benchmarks.md",
            "../reference/agentic-components.md",
        ):
            self.assertIn(f"({target})", reference)
        agentic = (SITE / "docs/benchmarks/creating-agentic-benchmarks.md").read_text()
        self.assertIn("(./creating-benchmarks.md#limits)", agentic)
        self.assertIn("(./creating-benchmarks.md#read-only-workspaces)", agentic)
        environment = (SITE / "docs/configuration/environment-configs.md").read_text()
        self.assertIn("(./kubernetes-deployment.md)", environment)
        self.assertIn("(../reference/sandbox-runtime.md)", environment)

    def test_landing_results_show_complete_multi_model_comparison(self):
        text = LANDING.read_text()
        results = re.findall(
            r'\{\s*model: "([^"]+)",\s*bench: "([^"]+)",\s*'
            r"samples: (\d+),\s*score: ([\d.]+),?\s*\}",
            text,
        )
        self.assertGreaterEqual(len(results), 6)
        models = {row[0] for row in results}
        benchmarks = {row[1] for row in results}
        self.assertGreaterEqual(len(models), 3)
        self.assertGreaterEqual(len(benchmarks), 2)
        pairs = {(row[0], row[1]) for row in results}
        self.assertEqual(len(pairs), len(results))
        self.assertEqual(pairs, {(m, b) for m in models for b in benchmarks})
        self.assertIn("new Set(EVALUATION_RESULTS.map((r) => r.model)).size", text)
        self.assertIn("new Set(EVALUATION_RESULTS.map((r) => r.bench)).size", text)
        for _, _, samples, score in results:
            self.assertGreater(int(samples), 0)
            self.assertGreaterEqual(float(score), 0)
            self.assertLessEqual(float(score), 1)
            # One-trial MCQA examples use an integral count of correct answers.
            correct = float(score) * int(samples)
            self.assertAlmostEqual(correct, round(correct))
        self.assertNotIn("boardRank", text)
        self.assertNotIn("Illustrative leaderboard", text)
        self.assertIn('aria-labelledby="evaluation-results-title"', text)
        self.assertIn('aria-describedby="evaluation-results-note"', text)
        self.assertIn('aria-label="Example evaluation results"', text)
        self.assertIn("tabIndex={0}", text)

    def test_deployment_override_does_not_change_experiment_identity(self):
        store = Mock()
        make_experiment = functools.partial(
            BenchmarkExperiment,
            benchmark_path=REPO
            / "src/faith/__benchmarks__/vulnerability-localization-a",
            generation_mode=GenerationMode.CHAT_COMP,
            prompt_format=PromptFormatter.CHAT,
            n_shot=SampleRatio(0),
            model_name="fixture",
            gen_params=GenParams(),
            root_datastore=store,
            num_trials=1,
            initial_seed=373363,
        )
        authored = make_experiment()
        deployment = make_experiment(
            environment_config=AgenticEnvironmentConfig(
                type="kubernetes", config={"warmpool": "fixture"}
            ),
        )
        first, second = store.sub_store.call_args_list
        self.assertEqual(first.args, second.args)
        self.assertTrue(first.args[0].name.startswith("agentic_"))
        assert authored.benchmark_config.agentic_config is not None
        assert deployment.benchmark_config.agentic_config is not None
        self.assertNotEqual(
            authored.benchmark_config.agentic_config.environment,
            deployment.benchmark_config.agentic_config.environment,
        )
        storage = (SITE / "docs/analytics/storage.md").read_text()
        self.assertIn("does not change that hash", storage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
