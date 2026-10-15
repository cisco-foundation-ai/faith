# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for agentic benchmark config types.

These types are the yaml-facing surface of an agentic benchmark, so what is
tested here is parse/serialize fidelity and the defaults a yaml may omit. The
limit *rules* live in `validate_agent_limits` and are covered exhaustively by
tests/agentic/core/test_react_loop.py; only the delegation is asserted here.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from faith._internal.algo.hash import dict_sha256
from faith._types.config.agentic import (
    AgenticAnswerExtractorConfig,
    AgenticConfig,
    AgenticEnvironmentConfig,
    AgenticLimitsConfig,
    AgenticTaskLoaderConfig,
    resolve_environment,
    validate_authored_environment,
)
from faith._types.model.engine import REDACTED

# A benchmark.yaml `agentic_config` naming every section and limit, and the
# config it parses to. Round-tripped, so it doubles as the serialized form.
_FULL_AGENTIC_DICT = {
    "task_loader": {"type": "vulnerability_localization", "config": {"split": "test"}},
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
        "recover_on_environment_failure": False,
    },
}

_FULL_AGENTIC_CONFIG = AgenticConfig(
    task_loader=AgenticTaskLoaderConfig(
        type="vulnerability_localization", config={"split": "test"}
    ),
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
        recover_on_environment_failure=False,
    ),
)


def test_environment_secrets_are_redacted_without_changing_runtime_or_identity() -> (
    None
):
    raw = {
        "environment": {"API_KEY": "fake-key", "AUTH_TOKEN": "fake-token", "LANG": "C"},
        "credentials": {"arbitrary": "fake-credentials"},
        "nested": [{"password": "fake-password", "max_tokens": 20}],
        "agent_image": "example:latest",
    }
    original = deepcopy(raw)
    environment = AgenticEnvironmentConfig("docker", raw)
    config = replace(_FULL_AGENTIC_CONFIG, environment=environment)
    recorded: dict[str, Any] = config.to_dict()
    expected = {
        "environment": {"API_KEY": REDACTED, "AUTH_TOKEN": REDACTED, "LANG": "C"},
        "credentials": REDACTED,
        "nested": [{"password": REDACTED, "max_tokens": 20}],
        "agent_image": "example:latest",
    }
    assert recorded["environment"]["config"] == expected
    assert environment.to_dict()["config"] == expected
    assert "fake-key" not in config.to_json()
    assert "fake-key" not in repr(config)
    assert environment.config == raw == original

    # The hashing representation must remain exactly the pre-redaction shape.
    recorded["environment"]["config"] = original
    assert config.to_identity_dict() == recorded
    assert dict_sha256(config.to_identity_dict()) == dict_sha256(recorded)
    identity = config.to_identity_dict()
    identity["environment"]["config"]["environment"]["API_KEY"] = "other"
    assert environment.config == original


def test_identity_without_environment_settings_keeps_the_original_shape() -> None:
    config = replace(
        _FULL_AGENTIC_CONFIG, environment=AgenticEnvironmentConfig("docker")
    )
    assert config.to_identity_dict() == config.to_dict()


def test_environment_assignments_preserve_runtime_and_identity() -> None:
    """Docker accepts NAME=value lists as well as environment mappings."""
    variables = [
        "API_KEY=fake-key=a=b",
        "AUTH_TOKEN=fake-token",
        "PASSWORD=",
        "LANG=C",
        "MAX_TOKENS=20",
        "STOP_TOKEN_IDS=1,2",
        "EMPTY=",
        "API_KEY",
        "DESCRIPTION=contains=a=b",
    ]
    expected = [
        f"API_KEY={REDACTED}",
        f"AUTH_TOKEN={REDACTED}",
        f"PASSWORD={REDACTED}",
        *variables[3:],
    ]
    raw: dict[str, Any] = {
        "environment": variables,
        "nested": [{"environment": variables}],
    }
    original = deepcopy(raw)
    environment = AgenticEnvironmentConfig("docker", raw)
    config = replace(_FULL_AGENTIC_CONFIG, environment=environment)
    recorded: dict[str, Any] = config.to_dict()

    assert recorded["environment"]["config"] == {
        "environment": expected,
        "nested": [{"environment": expected}],
    }
    assert "fake-key" not in config.to_json()
    assert "fake-token" not in environment.to_json()
    assert environment.config == original
    recorded["environment"]["config"] = original
    assert config.to_identity_dict() == recorded
    assert dict_sha256(config.to_identity_dict()) == dict_sha256(recorded)

    other = deepcopy(original)
    other["environment"][0] = "API_KEY=another-fake-key"
    other_config = replace(
        config, environment=AgenticEnvironmentConfig("docker", other)
    )
    assert dict_sha256(config.to_identity_dict()) != dict_sha256(
        other_config.to_identity_dict()
    )
    identity = config.to_identity_dict()
    identity["environment"]["config"]["environment"][0] = "API_KEY=changed"
    assert environment.config == original


# The minimal legal section: only the three implementation choices, everything
# else defaulted.
_MINIMAL_AGENTIC_DICT = {
    "task_loader": {"type": "vulnerability_localization"},
    "environment": {"type": "docker"},
    "answer_extractor": {"type": "file_list"},
}


def test_a_full_agentic_section_round_trips_through_a_dict() -> None:
    """A benchmark.yaml section must survive parse -> serialize (experiment.json)."""
    assert AgenticConfig.from_dict(_FULL_AGENTIC_DICT) == _FULL_AGENTIC_CONFIG
    assert _FULL_AGENTIC_CONFIG.to_dict() == _FULL_AGENTIC_DICT


@pytest.mark.parametrize("value", [None, "true", "false", 0, 1, [], {}])
def test_read_only_workspace_requires_a_boolean(value) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        AgenticConfig.from_dict({**_MINIMAL_AGENTIC_DICT, "read_only_workspace": value})


def test_read_only_workspace_changes_identity_without_changing_old_defaults() -> None:
    writable = AgenticConfig.from_dict(_MINIMAL_AGENTIC_DICT).to_dict()
    explicit_false = AgenticConfig.from_dict(
        {
            **_MINIMAL_AGENTIC_DICT,
            "read_only_workspace": False,
        }
    ).to_dict()
    protected = AgenticConfig.from_dict(
        {
            **_MINIMAL_AGENTIC_DICT,
            "read_only_workspace": True,
        }
    ).to_dict()
    assert writable == explicit_false
    assert "read_only_workspace" not in writable
    assert protected["read_only_workspace"] is True
    assert dict_sha256(protected) != dict_sha256(writable)


def test_existing_positional_agentic_configuration_remains_compatible() -> None:
    config = AgenticConfig(
        _FULL_AGENTIC_CONFIG.task_loader,
        _FULL_AGENTIC_CONFIG.environment,
        _FULL_AGENTIC_CONFIG.answer_extractor,
        "repo",
    )
    assert config.working_subdir == "repo"
    assert config.read_only_workspace is False


def test_the_omitted_sections_default_and_serialize_back_out() -> None:
    """A yaml naming only the three choices still records the limits it ran under.

    Empty `config`/`tools` and an absent `system_prompt` are dropped on the way out
    so a minimal yaml does not grow noise, but the limits are written explicitly: a
    run's record must say which budgets applied rather than leaving them to the next
    release's defaults.
    """
    config = AgenticConfig.from_dict(_MINIMAL_AGENTIC_DICT)

    assert config == AgenticConfig(
        task_loader=AgenticTaskLoaderConfig(type="vulnerability_localization"),
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
            recover_on_environment_failure=True,
        ),
    )
    assert config.to_dict() == _MINIMAL_AGENTIC_DICT | {
        "limits": {
            "max_tool_calls": 50,
            "max_turns": 100,
            "max_consecutive_no_tool_calls": 3,
            "command_timeout_seconds": 60,
            "recover_on_environment_failure": True,
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


def _agentic_config(environment_type: str, tools: list[str]) -> AgenticConfig:
    return AgenticConfig(
        task_loader=AgenticTaskLoaderConfig(type="vulnerability_localization"),
        environment=AgenticEnvironmentConfig(type=environment_type),
        answer_extractor=AgenticAnswerExtractorConfig(type="file_list"),
        tools=tools,
    )


def test_agentic_config_accepts_tools_the_environment_serves() -> None:
    assert _agentic_config("docker", ["bash_exec", "submit"]).tools == [
        "bash_exec",
        "submit",
    ]


def test_agentic_config_rejects_an_unregistered_environment() -> None:
    # Parse-time, so `faith list-benchmarks` surfaces a typo rather than an eval.
    with pytest.raises(ValueError, match="not a valid AgenticEnvironmentType"):
        _agentic_config("dokker", [])


def test_agentic_config_rejects_an_unregistered_tool() -> None:
    with pytest.raises(ValueError, match="not a valid AgenticToolType"):
        _agentic_config("docker", ["no_such_tool"])


def test_agentic_config_tolerates_a_bare_tools_key() -> None:
    """`tools:` with no value decodes to None, not to the field default."""
    config = AgenticConfig.from_dict(
        {
            "task_loader": {"type": "vulnerability_localization"},
            "environment": {"type": "docker"},
            "answer_extractor": {"type": "file_list"},
            "tools": None,
        }
    )
    assert config.tools == []


def test_an_environment_tolerates_a_bare_config_key() -> None:
    """`config:` with no value decodes to None too, and a null has no keys to read."""
    assert AgenticEnvironmentConfig.from_dict(
        {"type": "docker", "config": None}
    ) == AgenticEnvironmentConfig(type="docker", config={})


def test_a_bare_config_merges_as_empty_from_either_side() -> None:
    """Both merge branches read `config`, so a null on either side must be empty."""
    building = AgenticEnvironmentConfig(
        type="docker", config={"agent_dockerfile": "/b/Dockerfile"}
    )
    bare = AgenticEnvironmentConfig.from_dict({"type": "docker", "config": None})

    assert (
        resolve_environment(building, bare),
        resolve_environment(bare, building),
    ) == (building, building)


def test_an_authored_config_may_not_both_build_and_pull_its_image() -> None:
    """The one config shape the derived-tag scheme has no answer for.

    Naming an image *and* a Dockerfile asks for a local build under a name the
    author chose, which is the thing being removed. Rejected at parse, naming the
    file, because by the time an image exists under that tag neither the overwrite
    nor the stale reuse can be undone.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_authored_environment(
            {
                "agent_dockerfile": "$BENCHMARKS_ROOT/vulnerability-localization/Dockerfile",
                "agent_image": "faith-vulnerability-localization:latest",
            },
            "Benchmark config 'vulnerability-localization-template.yaml'",
        )

    assert str(excinfo.value) == (
        "Benchmark config 'vulnerability-localization-template.yaml' sets both 'agent_dockerfile' and "
        "'agent_image'. Set only 'agent_dockerfile' to build the agent image (FAITH "
        "names what it builds, from the build context's digest), or only 'agent_image' "
        "to pull a published one."
    )


@pytest.mark.parametrize(
    "config",
    [
        {"agent_dockerfile": "$BENCHMARKS_ROOT/vulnerability-localization/Dockerfile"},
        {"agent_image": "registry.example/vulnerability-localization:v3"},
        {"memory_limit": "4g"},
        {"agent_dockerfile": "/tmp/Dockerfile", "agent_image": ""},
    ],
    ids=["build_only", "pull_only", "neither_key", "one_key_left_empty"],
)
def test_an_authored_config_naming_at_most_one_of_the_two_is_accepted(
    config: dict[str, Any],
) -> None:
    """Including an empty override: yaml's `agent_image:` means unset, not "".

    An environment section may also name neither and inherit both from a
    deployment, so absence is not this function's business.
    """
    validate_authored_environment(config, "Environment config 'deployment.yaml'")


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"command_timeout_seconds": None},
        # 0 is `timeout`'s own "no bound", so it is a value rather than a mistake.
        {"command_timeout_seconds": 0},
        {"command_timeout_seconds": 120},
        {"command_timeout_seconds": "120"},
    ],
    ids=["absent", "explicit_null", "unbounded", "bounded", "from_yaml_string"],
)
def test_an_authored_config_naming_a_non_negative_bound_is_accepted(
    config: dict[str, Any],
) -> None:
    """Absence and 0 both pass: only a negative is unusable."""
    validate_authored_environment(config, "Environment config 'deployment.yaml'")


def test_an_authored_config_may_not_name_an_unknown_connection_mode() -> None:
    """A typo here otherwise surfaces as the first task's sandbox failing to connect.

    The mode is a deployment's business — whether FAITH runs inside the cluster — so
    the file it names is the one an operator wrote, not the benchmark.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_authored_environment(
            {"connection": "in-cluster"}, "Environment config 'deployment.yaml'"
        )

    assert str(excinfo.value) == (
        "Environment config 'deployment.yaml' sets 'connection' to 'in-cluster'; "
        "must be one of in_cluster, tunnel."
    )


@pytest.mark.parametrize(
    "config",
    [{}, {"connection": None}, {"connection": "in_cluster"}, {"connection": "tunnel"}],
    ids=["absent", "explicit_null", "in_cluster", "tunnel"],
)
def test_an_authored_config_naming_a_known_connection_mode_is_accepted(
    config: dict[str, Any],
) -> None:
    """Absence passes: an unset key means the environment's own default."""
    validate_authored_environment(config, "Environment config 'deployment.yaml'")


# A benchmark's own environment section: the portable docker default, with a key a
# deployment is expected to override, one only the benchmark knows, and a nested
# dict, which merges by a different rule than the scalars around it.
_BENCHMARK_ENVIRONMENT = AgenticEnvironmentConfig(
    type="docker",
    config={
        "agent_image": "faith-vulnerability-localization:latest",
        "memory_limit": "4g",
        "environment": {"LANG": "C.UTF-8", "BENCHMARK_SEED": "7"},
    },
)


@pytest.mark.parametrize(
    ("deployment", "expected"),
    [
        # The common case: no override, so the benchmark runs as authored.
        (None, _BENCHMARK_ENVIRONMENT),
        # Same type: the dicts merge, the deployment winning on the shared key
        # while the benchmark's own keys survive.
        (
            AgenticEnvironmentConfig(
                type="docker",
                config={
                    "agent_image": "registry.example/vulnerability-localization:v3"
                },
            ),
            AgenticEnvironmentConfig(
                type="docker",
                config={
                    "agent_image": "registry.example/vulnerability-localization:v3",
                    "memory_limit": "4g",
                    "environment": {"LANG": "C.UTF-8", "BENCHMARK_SEED": "7"},
                },
            ),
        ),
        # A nested dict merges rather than being replaced: a deployment adding one
        # container env var keeps the benchmark's, which it never mentioned.
        (
            AgenticEnvironmentConfig(
                type="docker",
                config={"environment": {"HTTPS_PROXY": "http://proxy.corp:3128"}},
            ),
            AgenticEnvironmentConfig(
                type="docker",
                config={
                    "agent_image": "faith-vulnerability-localization:latest",
                    "memory_limit": "4g",
                    "environment": {
                        "LANG": "C.UTF-8",
                        "BENCHMARK_SEED": "7",
                        "HTTPS_PROXY": "http://proxy.corp:3128",
                    },
                },
            ),
        ),
        # The deployment still wins on a nested key they both set.
        (
            AgenticEnvironmentConfig(
                type="docker", config={"environment": {"LANG": "en_US.UTF-8"}}
            ),
            AgenticEnvironmentConfig(
                type="docker",
                config={
                    "agent_image": "faith-vulnerability-localization:latest",
                    "memory_limit": "4g",
                    "environment": {"LANG": "en_US.UTF-8", "BENCHMARK_SEED": "7"},
                },
            ),
        ),
        # A type switch replaces the dict: `agent_image` and `memory_limit` name
        # things a cluster cannot read, so carrying them over would be noise the
        # backend silently ignores. Recursion must not reach across the switch and
        # revive the benchmark's nested keys either.
        (
            AgenticEnvironmentConfig(
                type="kubernetes",
                config={
                    "warmpool": "vulnerability-localization-pool",
                    "environment": {"KUBECONFIG": "/etc/kube"},
                },
            ),
            AgenticEnvironmentConfig(
                type="kubernetes",
                config={
                    "warmpool": "vulnerability-localization-pool",
                    "environment": {"KUBECONFIG": "/etc/kube"},
                },
            ),
        ),
        # A bare type switch with no config of its own still drops the docker keys.
        (
            AgenticEnvironmentConfig(type="kubernetes"),
            AgenticEnvironmentConfig(type="kubernetes"),
        ),
        # The registry reads `type` case-insensitively, so a differently-cased
        # spelling is the same backend: it merges, and is recorded lowercased.
        (
            AgenticEnvironmentConfig(type="Docker", config={"memory_limit": "16g"}),
            AgenticEnvironmentConfig(
                type="docker",
                config={
                    "agent_image": "faith-vulnerability-localization:latest",
                    "memory_limit": "16g",
                    "environment": {"LANG": "C.UTF-8", "BENCHMARK_SEED": "7"},
                },
            ),
        ),
        # A genuine switch is recorded canonically too, so the run's record names
        # one backend one way.
        (
            AgenticEnvironmentConfig(type="KUBERNETES", config={"warmpool": "vl"}),
            AgenticEnvironmentConfig(type="kubernetes", config={"warmpool": "vl"}),
        ),
    ],
    ids=[
        "no_override",
        "same_type_merges",
        "nested_dict_merges",
        "nested_key_overridden",
        "type_switch_replaces",
        "bare_type_switch",
        "case_insensitive_same_type",
        "case_insensitive_type_switch",
    ],
)
def test_a_deployment_environment_resolves_over_the_benchmarks(
    deployment: AgenticEnvironmentConfig | None,
    expected: AgenticEnvironmentConfig,
) -> None:
    assert resolve_environment(_BENCHMARK_ENVIRONMENT, deployment) == expected


# A benchmark that ships a Dockerfile and lets FAITH name what it builds, which is
# what a deployment's `agent_image` has to be able to override.
_BUILDING_BENCHMARK = AgenticEnvironmentConfig(
    type="docker",
    config={
        "agent_dockerfile": "$BENCHMARKS_ROOT/vulnerability-localization/Dockerfile",
        "memory_limit": "4g",
        "environment": {"LANG": "C.UTF-8"},
    },
)


@pytest.mark.parametrize(
    ("deployment_config", "expected_config"),
    [
        # A deployment naming an image asks to pull it instead of building. Both keys
        # stay in the merged config; which one wins is the environment's reading of
        # the pair, not something the merge rearranges.
        (
            {"agent_image": "ubuntu:24.04"},
            {
                "agent_dockerfile": "$BENCHMARKS_ROOT/vulnerability-localization/Dockerfile",
                "agent_image": "ubuntu:24.04",
                "memory_limit": "4g",
                "environment": {"LANG": "C.UTF-8"},
            },
        ),
        # Overriding something else entirely leaves the benchmark building its own.
        (
            {"memory_limit": "8g"},
            {
                "agent_dockerfile": "$BENCHMARKS_ROOT/vulnerability-localization/Dockerfile",
                "memory_limit": "8g",
                "environment": {"LANG": "C.UTF-8"},
            },
        ),
        # An image override beside a nested merge: no key here is special, so the
        # nested dict merges exactly as it does anywhere else.
        (
            {
                "agent_image": "ubuntu:24.04",
                "environment": {"HTTPS_PROXY": "http://proxy.corp:3128"},
            },
            {
                "agent_dockerfile": "$BENCHMARKS_ROOT/vulnerability-localization/Dockerfile",
                "agent_image": "ubuntu:24.04",
                "memory_limit": "4g",
                "environment": {
                    "LANG": "C.UTF-8",
                    "HTTPS_PROXY": "http://proxy.corp:3128",
                },
            },
        ),
    ],
    ids=[
        "image_override_keeps_both_keys",
        "unrelated_key_keeps_the_dockerfile",
        "image_override_with_nested_merge",
    ],
)
def test_a_deployment_image_merges_over_a_building_benchmark_without_a_special_case(
    deployment_config: dict[str, Any], expected_config: dict[str, Any]
) -> None:
    """No key-specific rule survives here: `agent_image` merges like anything else.

    The previous design deleted the inherited `agent_dockerfile` during the merge,
    because the built tag was whatever the config named and the two keys together
    were ambiguous. Deriving the built tag removes the ambiguity, so precedence moves
    to where the keys are read and the merge goes back to being generic.
    """
    resolved = resolve_environment(
        _BUILDING_BENCHMARK,
        AgenticEnvironmentConfig(type="docker", config=deployment_config),
    )
    assert resolved == AgenticEnvironmentConfig(type="docker", config=expected_config)


def test_a_benchmarks_own_type_is_read_case_insensitively_too() -> None:
    """The benchmark.yaml is as free to capitalize `type` as the deployment file is.

    Both sides go through the registry, so neither spelling decides the merge.
    """
    resolved = resolve_environment(
        AgenticEnvironmentConfig(type="Docker", config={"agent_image": "bench:v1"}),
        AgenticEnvironmentConfig(type="docker", config={"memory_limit": "16g"}),
    )

    assert resolved == AgenticEnvironmentConfig(
        type="docker", config={"agent_image": "bench:v1", "memory_limit": "16g"}
    )


def test_resolving_leaves_the_benchmarks_own_section_untouched() -> None:
    """The merge must not mutate the loaded config: `config` is a shared dict.

    Overriding a nested key too, since the recursive merge writes into its source:
    a shallow copy of `config` would leave the nested dict shared and still be
    caught here.
    """
    resolve_environment(
        _BENCHMARK_ENVIRONMENT,
        AgenticEnvironmentConfig(
            type="docker",
            config={
                "agent_image": "other:v2",
                "environment": {"LANG": "en_US.UTF-8", "HTTPS_PROXY": "p"},
            },
        ),
    )
    assert _BENCHMARK_ENVIRONMENT.config == {
        "agent_image": "faith-vulnerability-localization:latest",
        "memory_limit": "4g",
        "environment": {"LANG": "C.UTF-8", "BENCHMARK_SEED": "7"},
    }


def test_a_resolved_environment_round_trips_through_a_dict() -> None:
    """`eval` reads the resolved section back out of the experiment summary."""
    resolved = resolve_environment(
        _BENCHMARK_ENVIRONMENT,
        AgenticEnvironmentConfig(
            type="kubernetes", config={"warmpool": "vulnerability-localization"}
        ),
    )
    assert AgenticEnvironmentConfig.from_dict(resolved.to_dict()) == resolved
