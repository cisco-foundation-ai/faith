# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the argument surface `faith`'s sub-commands expose.

Only the flags this module owns are asserted; what the sub-commands then do with
them belongs to tests/cli/subcmd.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from faith._types.config.agentic import AgenticEnvironmentConfig

_MINIMAL_ARGS = [
    "--datastore-location",
    "/tmp/faith-test-datastore",
    "--model-paths",
    "test-model",
    "--prompt-format",
    "chat",
]

# `faith.cli.cli` parses sys.argv as it is imported, so the import needs an argv
# that parses. Nothing else imports it, so this is where it is first loaded.
with patch.object(sys, "argv", ["faith", "query", *_MINIMAL_ARGS]):
    from faith.cli.cli import _cli_parser, _cli_query

_ENVIRONMENT_CONFIG = (
    Path(__file__).parents[1]
    / "benchmark"
    / "testdata"
    / "env-config"
    / "deployment-kubernetes.yaml"
)


@pytest.mark.parametrize(
    ("subcommand", "extra_args", "accepted"),
    [
        ("query", _MINIMAL_ARGS, True),
        # run-all shares one args namespace with query, so it gets the flag too.
        ("run-all", _MINIMAL_ARGS, True),
        # Standalone eval reads the already-resolved environment out of the
        # experiment summary, so choosing one is not its to do.
        ("eval", ["--experiment-path", "/tmp/faith-test-experiment"], False),
    ],
    ids=["query", "run_all", "eval"],
)
def test_which_subcommands_accept_an_environment_config(
    subcommand: str, extra_args: list[str], accepted: bool
) -> None:
    args = [subcommand, "--environment-config", str(_ENVIRONMENT_CONFIG), *extra_args]
    if not accepted:
        with pytest.raises(SystemExit):
            _cli_parser.parse_args(args)
        return
    assert _cli_parser.parse_args(args).environment_config == _ENVIRONMENT_CONFIG


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["--environment-config", str(_ENVIRONMENT_CONFIG)],
            AgenticEnvironmentConfig(
                type="kubernetes",
                config={
                    "warmpool": "vulnerability-localization-pool",
                    "namespace": "sandbox-pool",
                },
            ),
        ),
        ([], None),
    ],
    ids=["given", "omitted"],
)
def test_an_environment_config_path_is_loaded_before_the_run(
    args: list[str], expected: AgenticEnvironmentConfig | None
) -> None:
    """The flag is a path; what the run needs is the parsed config, loaded once."""
    namespace = _cli_parser.parse_args(["query", *args, *_MINIMAL_ARGS])
    with patch("faith.cli.subcmd.query.run_experiment_queries") as run:
        _cli_query(namespace, Mock())

    assert run.call_args.args[1].environment_config == expected
