# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for agentic orchestration wiring.

The production registries are empty until their implementations land, so these
tests define fake subclasses of the shared resolution bases with fake members.
An Enum with members cannot be subclassed, so the fakes derive from the empty
bases the production registries derive from, exercising the same `create` code.
"""

from unittest.mock import patch

import pytest

from faith._types.config.agentic import (
    AgenticAnswerExtractorConfig,
    AgenticConfig,
    AgenticEnvironmentConfig,
    AgenticTaskLoaderConfig,
)
from faith._types.json_types import JsonValue
from faith._types.record.stats import Labeling
from faith.agentic.core import (
    AnswerExtractor,
    Environment,
    TaskLoader,
    Tool,
)
from faith.agentic.orchestration import (
    _ConfiguredRegistry,
    _NoConfigRegistry,
    _WorkingDirRegistry,
    build_answer_extractor,
    build_environment,
    build_task_loader,
    build_tool_registry,
)


class _FakeLoader(TaskLoader):
    """Loader double that records what the base class was constructed with."""

    @property
    def received(self) -> tuple[dict, str]:
        """Expose the config and working directory the base class stored."""
        return self._config, self._working_dir

    def load_tasks(self, benchmark_path):
        return []


class _FakeEnvironment(Environment):
    """Environment double that records the config it was constructed with."""

    def __init__(self, config: dict | None = None):
        self.config = config

    def setup(self, task):
        pass

    def teardown(self):
        pass


class _FakeExtractor(AnswerExtractor):
    """Extractor double that records the config it was constructed with."""

    @property
    def received_config(self) -> dict:
        """Expose the config the base class stored."""
        return self._config

    def extract(self, tool_name: str | None, answer: JsonValue) -> Labeling | None:
        return None


class _FakeTool(Tool):
    """Tool double with a fixed name and no-op execution."""

    @property
    def name(self) -> str:
        return "fake_tool"

    @property
    def description(self) -> str:
        return "A fake tool."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def _invoke(self, environment, args):
        return "ok"


class _FakeLoaderType(_WorkingDirRegistry):
    FAKE = (_FakeLoader,)


class _FakeEnvType(_ConfiguredRegistry):
    FAKE = (_FakeEnvironment,)


class _FakeExtractorType(_ConfiguredRegistry):
    FAKE = (_FakeExtractor,)


class _FakeToolType(_NoConfigRegistry):
    FAKE_TOOL = (_FakeTool,)


def _agentic_config(**overrides) -> AgenticConfig:
    """An agentic config naming the fake types, with the section under test replaced."""
    return AgenticConfig(
        **{
            "task_loader": AgenticTaskLoaderConfig(type="fake"),
            "environment": AgenticEnvironmentConfig(type="fake"),
            "answer_extractor": AgenticAnswerExtractorConfig(type="fake"),
            **overrides,
        }
    )


def test_build_task_loader_passes_its_config_and_the_working_dir() -> None:
    """A loader is told both its own config and where the agent will work.

    The directory comes from the top-level `working_subdir`, not from the loader's
    config, so a prompt naming it cannot name a directory the environment isn't
    using.
    """
    with patch("faith.agentic.orchestration.AgenticTaskLoaderType", _FakeLoaderType):
        loader = build_task_loader(
            _agentic_config(
                task_loader=AgenticTaskLoaderConfig(type="fake", config={"depth": 2}),
                working_subdir="repo",
            )
        )
    assert isinstance(loader, _FakeLoader)
    assert loader.received == ({"depth": 2}, "/workspace/repo")


def test_build_environment_injects_the_working_subdir_into_its_config() -> None:
    """The subdir reaches the environment through its config, alongside its own keys.

    `Environment` declares only a lifecycle, so the one path it needs arrives the
    same way the rest of its settings do rather than as a constructor argument.
    """
    with patch("faith.agentic.orchestration.AgenticEnvironmentType", _FakeEnvType):
        env = build_environment(
            _agentic_config(
                environment=AgenticEnvironmentConfig(
                    type="fake", config={"image": "ubuntu:22.04"}
                ),
                working_subdir="repo",
            )
        )
    assert isinstance(env, _FakeEnvironment)
    assert env.config == {"image": "ubuntu:22.04", "working_subdir": "repo"}


def test_an_environment_config_cannot_override_the_working_subdir() -> None:
    """A stale `working_subdir` under `environment` loses to the top-level key.

    The duplication this replaces is exactly a benchmark carrying two answers, so
    the authoritative one has to win rather than be silently shadowed.
    """
    with patch("faith.agentic.orchestration.AgenticEnvironmentType", _FakeEnvType):
        env = build_environment(
            _agentic_config(
                environment=AgenticEnvironmentConfig(
                    type="fake", config={"working_subdir": "stale"}
                ),
                working_subdir="repo",
            )
        )
    assert isinstance(env, _FakeEnvironment)
    assert env.config == {"working_subdir": "repo"}


def test_build_answer_extractor_resolves_and_constructs() -> None:
    """build_answer_extractor resolves the extractor by name and passes its config."""
    with patch(
        "faith.agentic.orchestration.AgenticAnswerExtractorType", _FakeExtractorType
    ):
        extractor = build_answer_extractor(
            AgenticAnswerExtractorConfig(type="fake", config={"strict": True})
        )
    assert isinstance(extractor, _FakeExtractor)
    assert extractor.received_config == {"strict": True}


def test_a_config_naming_an_unregistered_type_is_rejected() -> None:
    """An unregistered short name fails at build time, naming the bad value."""
    with patch("faith.agentic.orchestration.AgenticEnvironmentType", _FakeEnvType):
        with pytest.raises(ValueError, match="nope"):
            build_environment(
                _agentic_config(environment=AgenticEnvironmentConfig(type="nope"))
            )


@pytest.mark.parametrize(
    ("tool_names", "expected"),
    [([], []), (["fake_tool"], ["fake_tool"])],
    ids=["empty", "one_tool"],
)
def test_build_tool_registry_registers_named_tools(
    tool_names: list[str], expected: list[str]
) -> None:
    """build_tool_registry resolves each name and registers the tool instance.

    Tools take no config, so resolution here is a name lookup plus an
    argument-free construction; the registry keys them by the tool's own `name`.
    """
    with patch("faith.agentic.orchestration.AgenticToolType", _FakeToolType):
        registry = build_tool_registry(tool_names)
    assert registry.list_tools() == expected
