# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for agentic benchmark config types.

These types are the yaml-facing surface of an agentic benchmark, so what is
tested here is parse/serialize fidelity and the defaults a yaml may omit. The
limit *rules* live in `validate_agent_limits` and are covered exhaustively by
tests/agentic/core/test_react_loop.py; only the delegation is asserted here.
"""

import pytest

from faith._types.config.agentic import (
    AgenticAnswerExtractorConfig,
    AgenticConfig,
    AgenticEnvironmentConfig,
    AgenticLimitsConfig,
    AgenticTaskLoaderConfig,
)

# A benchmark.yaml `agentic_config` naming every section and limit, and the
# config it parses to. Round-tripped, so it doubles as the serialized form.
_FULL_AGENTIC_DICT = {
    "task_loader": {"type": "vulnloc", "config": {"split": "test"}},
    "environment": {"type": "docker", "config": {"image": "ubuntu:22.04"}},
    "answer_extractor": {"type": "file_list", "config": {"strip_prefix": "src/"}},
    "working_subdir": "repo",
    "tools": ["bash_exec", "submit"],
    "system_prompt": "You are a security researcher. Find the crashing input.",
    "limits": {
        "max_tool_calls": 20,
        "max_turns": 40,
        "max_consecutive_no_tool_calls": 2,
        "timeout_seconds": 900.0,
        "command_timeout_seconds": 10,
    },
}

_FULL_AGENTIC_CONFIG = AgenticConfig(
    task_loader=AgenticTaskLoaderConfig(type="vulnloc", config={"split": "test"}),
    environment=AgenticEnvironmentConfig(
        type="docker", config={"image": "ubuntu:22.04"}
    ),
    answer_extractor=AgenticAnswerExtractorConfig(
        type="file_list", config={"strip_prefix": "src/"}
    ),
    working_subdir="repo",
    tools=["bash_exec", "submit"],
    system_prompt="You are a security researcher. Find the crashing input.",
    limits=AgenticLimitsConfig(
        max_tool_calls=20,
        max_turns=40,
        max_consecutive_no_tool_calls=2,
        timeout_seconds=900.0,
        command_timeout_seconds=10,
    ),
)

# The minimal legal section: only the three implementation choices, everything
# else defaulted.
_MINIMAL_AGENTIC_DICT = {
    "task_loader": {"type": "vulnloc"},
    "environment": {"type": "docker"},
    "answer_extractor": {"type": "file_list"},
}


def test_a_full_agentic_section_round_trips_through_a_dict() -> None:
    """A benchmark.yaml section must survive parse -> serialize (experiment.json)."""
    assert AgenticConfig.from_dict(_FULL_AGENTIC_DICT) == _FULL_AGENTIC_CONFIG
    assert _FULL_AGENTIC_CONFIG.to_dict() == _FULL_AGENTIC_DICT


def test_the_omitted_sections_default_and_serialize_back_out() -> None:
    """A yaml naming only the three choices still records the limits it ran under.

    Empty `config`/`tools` and an absent `system_prompt` are dropped on the way out
    so a minimal yaml does not grow noise, but the limits are written explicitly: a
    run's record must say which budgets applied rather than leaving them to the next
    release's defaults.
    """
    config = AgenticConfig.from_dict(_MINIMAL_AGENTIC_DICT)

    assert config == AgenticConfig(
        task_loader=AgenticTaskLoaderConfig(type="vulnloc"),
        environment=AgenticEnvironmentConfig(type="docker"),
        answer_extractor=AgenticAnswerExtractorConfig(type="file_list"),
        working_subdir="",
        tools=[],
        system_prompt=None,
        limits=AgenticLimitsConfig(
            max_tool_calls=50,
            max_turns=100,
            max_consecutive_no_tool_calls=3,
            timeout_seconds=None,
            command_timeout_seconds=60,
        ),
    )
    assert config.to_dict() == _MINIMAL_AGENTIC_DICT | {
        "limits": {
            "max_tool_calls": 50,
            "max_turns": 100,
            "max_consecutive_no_tool_calls": 3,
            "command_timeout_seconds": 60,
        }
    }


def test_an_explicitly_empty_system_prompt_survives_the_round_trip() -> None:
    """An authored empty prompt is recorded as one, not collapsed into omission.

    `""` and `None` both send no system message, but they are different yaml: the
    exclusion is on `is None`, so a benchmark that deliberately blanks the prompt
    reads back as having blanked it rather than as never having set it.
    """
    blanked = _MINIMAL_AGENTIC_DICT | {"system_prompt": ""}
    config = AgenticConfig.from_dict(blanked)

    assert config.system_prompt == ""
    assert config.to_dict()["system_prompt"] == ""


@pytest.mark.parametrize(
    ("subdir", "working_dir"),
    [
        ("", "/workspace"),
        ("repo", "/workspace/repo"),
        ("build/out", "/workspace/build/out"),
        ("./repo", "/workspace/repo"),
        ("repo/", "/workspace/repo"),
    ],
)
def test_the_working_dir_is_derived_from_the_authored_subdir(
    subdir: str, working_dir: str
) -> None:
    """The absolute directory is joined onto the workspace, never authored.

    This is the single statement of where the agent works: the environment runs
    its commands there and a loader renders it into a prompt, so a benchmark
    naming it twice could not disagree with itself.
    """
    assert (
        AgenticConfig.from_dict(
            _MINIMAL_AGENTIC_DICT | {"working_subdir": subdir}
        ).working_dir
        == working_dir
    )


def test_a_bare_working_subdir_key_reads_as_no_subdir() -> None:
    """`working_subdir:` with no value decodes to None, not to the field default."""
    config = AgenticConfig.from_dict(_MINIMAL_AGENTIC_DICT | {"working_subdir": None})

    assert (config.working_subdir, config.working_dir) == ("", "/workspace")


@pytest.mark.parametrize("subdir", ["/etc", "..", "../etc", "repo/../.."])
def test_a_subdir_outside_the_workspace_is_rejected_at_parse(subdir: str) -> None:
    """An absolute or climbing subdir would move the agent out of the sandbox.

    Checked on the derived path, so both spellings of the escape are one rule.
    """
    with pytest.raises(ValueError) as excinfo:
        AgenticConfig.from_dict(_MINIMAL_AGENTIC_DICT | {"working_subdir": subdir})

    assert "does not name a directory inside the workspace" in str(excinfo.value)


def test_a_yaml_supplied_limit_is_validated_at_parse() -> None:
    """A misconfigured yaml fails at parse, before any model call.

    `AgenticLimitsConfig` shares `validate_agent_limits` with `ReActConfig`, whose
    tests own the rule table; this pins that a yaml goes through it at all.
    """
    with pytest.raises(ValueError) as excinfo:
        AgenticLimitsConfig.from_dict({"max_tool_calls": 50, "max_turns": 10})

    assert str(excinfo.value) == "max_turns (10) must be >= max_tool_calls (50)."


def test_a_turn_cap_equal_to_the_tool_budget_is_allowed() -> None:
    """The boundary is legal: every turn spending exactly one call is well-behaved."""
    assert AgenticLimitsConfig.from_dict(
        {"max_tool_calls": 50, "max_turns": 50}
    ) == AgenticLimitsConfig(max_tool_calls=50, max_turns=50)


@pytest.mark.parametrize("given", [1.5, 60], ids=["float", "int"])
def test_a_yaml_timeout_is_read_as_fractional_seconds(given: float) -> None:
    """yaml may write the timeout as an int, but it must not truncate to one.

    The loop compares it against `time.time()` deltas, so 1.5 has to survive.
    """
    assert AgenticLimitsConfig.from_dict(
        {"timeout_seconds": given}
    ).timeout_seconds == float(given)


def test_a_negative_command_bound_is_rejected_at_parse() -> None:
    """A negative reaches coreutils `timeout` as a flag, failing every command."""
    with pytest.raises(ValueError) as excinfo:
        AgenticLimitsConfig.from_dict({"command_timeout_seconds": -1})

    assert str(excinfo.value) == (
        "command_timeout_seconds (-1) must be non-negative; use 0 for no bound."
    )


def test_an_unbounded_command_is_written_as_zero() -> None:
    """0 is `timeout`'s own "no bound", so it must survive rather than default."""
    assert AgenticLimitsConfig.from_dict(
        {"command_timeout_seconds": 0}
    ) == AgenticLimitsConfig(command_timeout_seconds=0)
