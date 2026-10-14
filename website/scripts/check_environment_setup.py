#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Check documented setup examples without a daemon, cluster, or model endpoint.

Read the actual documentation snippets, parse their commands/configurations,
and exercise environment construction with SDK boundaries replaced by mocks.
Live deployment and image-build validation remain separate operator checks.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import yaml
from packaging.specifiers import SpecifierSet

from faith.agentic.environments.docker import DockerEnvironment
from faith.agentic.environments.kubernetes import KubernetesEnvironment
from faith.agentic.loaders._fetch import cache_root
from faith.agentic.orchestration import build_environment
from faith.benchmark.config import load_config_from_path, load_environment_config

# These tests inspect backend configuration without starting a real environment.
# pylint: disable=protected-access

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "website/docs"
ENV_PAGE = DOCS / "configuration/environment-configs.md"
K8S_PAGE = DOCS / "configuration/kubernetes-deployment.md"
BENCHMARK = REPO / "src/faith/__benchmarks__/vulnerability-localization-a"
TEMPLATE = (
    REPO / "src/faith/__benchmarks__/vulnerability-localization/sandbox-template.yaml"
)


def snippet(page: Path, title: str) -> str:
    """Return one uniquely titled code block from the documentation."""
    matches = re.findall(
        rf'```\w+ title="{re.escape(title)}"\n(.*?)```', page.read_text(), re.S
    )
    if len(matches) != 1:
        raise AssertionError(f"Expected one {title!r} example in {page}")
    return matches[0]


class EnvironmentSetupTests(unittest.TestCase):
    """Validate documented setup with SDK and service boundaries mocked."""

    # Keep independently reported setup cases together with their fixtures.
    # pylint: disable=too-many-public-methods

    def setUp(self):
        # unittest owns this context until cleanup, including on test failure.
        # pylint: disable-next=consider-using-with
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(
            patch("socket.create_connection", side_effect=AssertionError("network"))
        )
        self.enterContext(
            patch("subprocess.Popen", side_effect=AssertionError("subprocess"))
        )

    def deployment(self):
        """Load the deployment override published in the Kubernetes guide."""
        config_file = self.root / "k8s-environment.yaml"
        config_file.write_text(snippet(K8S_PAGE, "k8s-environment.yaml"))
        return load_environment_config(config_file)

    def test_installation_pins_a_supported_python_and_existing_extras(self):
        project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
        installation = (DOCS / "getting-started/installation.md").read_text()
        versions = re.findall(r"^uv venv --python (\S+) \.venv$", installation, re.M)
        self.assertGreaterEqual(len(versions), 2)
        for version in versions:
            self.assertIn(version, SpecifierSet(project["requires-python"]))
        self.assertNotRegex(installation, re.compile(r"^uv venv \.venv", re.M))
        for page in (installation, ENV_PAGE.read_text(), K8S_PAGE.read_text()):
            for extras in re.findall(r"\.\[([a-z,]+)\]", page):
                self.assertTrue(
                    set(extras.split(",")) <= project["optional-dependencies"].keys()
                )
        self.assertIn(
            "k8s-agent-sandbox==1.0.2", project["optional-dependencies"]["kubernetes"]
        )

    def run_docker_preflight(self, client):
        """Run the documented preflight against an injected SDK client."""
        example = snippet(ENV_PAGE, "Docker preflight")
        code = example.split("python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
        sdk = ModuleType("docker")
        from_env = Mock(return_value=client)
        sdk.__dict__["from_env"] = from_env
        output = StringIO()
        with patch.dict(sys.modules, {"docker": sdk}), redirect_stdout(output):
            # The code is a trusted, repository-owned documentation example.
            # pylint: disable-next=exec-used
            exec(compile(code, str(ENV_PAGE), "exec"), {})
        from_env.assert_called_once_with(timeout=10)
        return output.getvalue()

    def test_docker_preflight_uses_the_sdk_and_closes_client(self):
        client = Mock()
        client.ping.return_value = True
        self.assertIn("Docker daemon reachable", self.run_docker_preflight(client))
        client.ping.assert_called_once()
        client.close.assert_called_once()

    def test_docker_preflight_does_not_report_success_after_failed_ping(self):
        client = Mock()
        client.ping.return_value = False
        with self.assertRaisesRegex(AssertionError, "did not respond"):
            self.run_docker_preflight(client)
        client.close.assert_called_once()

    def test_packaged_docker_settings_match_documented_requirements(self):
        config = load_config_from_path(BENCHMARK).agentic_config
        assert config is not None
        with patch("faith.agentic.environments.docker._require_sdk"):
            env = build_environment(config)
        assert isinstance(env, DockerEnvironment)
        self.assertEqual(env._network, "none")
        self.assertEqual(env._memory_limit, "4g")
        self.assertEqual(env._cpu_limit, 2.0)
        self.assertEqual(env._hardening["pids_limit"], 512)
        self.assertEqual(env._hardening["cap_drop"], ["ALL"])
        self.assertEqual(env._hardening["security_opt"], ["no-new-privileges"])
        self.assertTrue(env._read_only_workspace)
        self.assertIsNone(env._client)
        self.assertRegex(ENV_PAGE.read_text(), r"\| `pids_limit` \| `512` \|")

    def test_documented_docker_override_preserves_image_and_workspace_policy(self):
        config_file = self.root / "docker-environment.yaml"
        config_file.write_text(snippet(ENV_PAGE, "docker-environment.yaml"))
        authored = load_config_from_path(BENCHMARK).agentic_config
        config = load_config_from_path(
            BENCHMARK, load_environment_config(config_file)
        ).agentic_config
        assert config is not None and authored is not None
        self.assertTrue(config.read_only_workspace)
        self.assertEqual(config.environment.config["memory_limit"], "8g")
        self.assertEqual(
            config.environment.config["agent_dockerfile"],
            authored.environment.config["agent_dockerfile"],
        )
        self.assertEqual(config.limits, authored.limits)

    def test_kubernetes_override_preserves_policy_and_configures_router(self):
        config = load_config_from_path(BENCHMARK, self.deployment()).agentic_config
        assert config is not None
        client, direct, tunnel = Mock(), Mock(), Mock()
        with patch(
            "faith.agentic.environments.kubernetes._require_sdk",
            return_value=(client, direct, tunnel),
        ):
            env = build_environment(config)
            assert isinstance(env, KubernetesEnvironment)
            client.assert_not_called()
            env._get_client()
        self.assertIsInstance(env, KubernetesEnvironment)
        self.assertTrue(env._read_only_workspace)
        self.assertEqual(env.working_dir, "/workspace/repo")
        self.assertEqual(env._command_timeout, 10)
        self.assertEqual(env._shutdown_after, 2400)
        self.assertNotIn("agent_dockerfile", config.environment.config)
        tunnel.assert_called_once_with(router_namespace="agent-sandbox-system")
        direct.assert_not_called()
        client.assert_called_once_with(
            connection_config=tunnel.return_value, cleanup=True
        )

    def test_kubernetes_defaults_and_explicit_ttl(self):
        env = KubernetesEnvironment({"warmpool": "fixture"}, client=Mock())
        self.assertEqual(env._connection, "in_cluster")
        self.assertEqual(env._router_namespace, "agent-sandbox-system")
        self.assertEqual(env._ready_timeout, 120)
        deployment = self.deployment()
        deployment.config["shutdown_after_seconds"] = 3000
        config = load_config_from_path(BENCHMARK, deployment).agentic_config
        assert config is not None
        with patch("faith.agentic.environments.kubernetes._require_sdk"):
            configured = build_environment(config)
            assert isinstance(configured, KubernetesEnvironment)
            self.assertEqual(configured._shutdown_after, 3000)

    def test_cluster_preflight_names_match_reference_resources(self):
        deployment = self.deployment().config
        template, pool = yaml.safe_load_all(TEMPLATE.read_text())
        self.assertEqual(pool["metadata"]["name"], deployment["warmpool"])
        self.assertEqual(pool["metadata"]["namespace"], deployment["namespace"])
        self.assertEqual(template["metadata"]["namespace"], deployment["namespace"])
        self.assertEqual(template["apiVersion"], "extensions.agents.x-k8s.io/v1beta1")
        self.assertEqual(
            pool["spec"]["sandboxTemplateRef"]["name"], template["metadata"]["name"]
        )
        commands = (
            snippet(K8S_PAGE, "Kubernetes preflight").replace("\\\n", "").splitlines()
        )
        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertEqual(shlex.split(command)[:2], ["kubectl", "get"])
        self.assertIn(deployment["warmpool"], commands[0])
        self.assertIn(template["metadata"]["name"], commands[1])
        self.assertIn(deployment["router_namespace"], commands[2])

    def test_cache_location_is_independent_of_result_storage(self):
        with patch.dict(
            os.environ, {"FAITH_CACHE_DIR": str(self.root), "XDG_CACHE_HOME": "/unused"}
        ):
            self.assertEqual(
                cache_root("vulnerability_localization"),
                self.root / "faith/vulnerability_localization",
            )
        with patch.dict(
            os.environ, {"FAITH_CACHE_DIR": "", "XDG_CACHE_HOME": str(self.root)}
        ):
            self.assertEqual(
                cache_root("vulnerability_localization"),
                self.root / "faith/vulnerability_localization",
            )

    def test_smoke_commands_parse_and_share_a_bounded_result_directory(self):
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
            # cli.py parses argv at import time; use the fixture above.
            # pylint: disable-next=import-outside-toplevel
            from faith.cli import cli
        page = DOCS / "getting-started/quickstart.md"
        query = snippet(page, "Agentic smoke query").split("\nfaith ", 1)[1]
        args = cli._cli_parser.parse_args(shlex.split(query.replace("\\\n", "")))
        self.assertEqual(args.sample_size, 1)
        self.assertEqual(args.num_trials, 1)
        self.assertEqual(str(args.model_engine), "openai")
        commands = (
            snippet(page, "Agentic smoke results").replace("\\\n", "").splitlines()
        )
        self.assertEqual(len(commands), 2)
        for command in commands:
            results = cli._cli_parser.parse_args(shlex.split(command)[1:])
            self.assertEqual(str(results.experiment_path), str(args.datastore_location))
        self.assertEqual(
            results.stats, ["mean_file_f1", "error_rate", "stopped_reason"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
