#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Check model documentation against config resolution and request construction.

No GPU, optional engine SDK, model download, or live endpoint is required.
The OpenAI SDK constructor is replaced, but FAITH's wrapper, CLI resolver,
Q&A transform, probe, and agentic request builder are exercised directly.
"""

from __future__ import annotations

import inspect
import re
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import yaml

from faith._internal.functools.retriable import (
    MaxAttemptsExceededError,
    RetryFunctionWrapper,
)
from faith._types.model.engine import ModelEngine
from faith._types.model.generation import GenerationMode, GenParams
from faith._types.model.prompt import PromptFormatter
from faith._types.model.spec import ModelSpec
from faith.agentic.core import ReActLoop, Task, ToolRegistry
from faith.agentic.core.react_loop import validate_agentic_generation_params
from faith.cli.subcmd.query import _check_tool_calling
from faith.model.config import load_model_config
from faith.model.listing import choice_to_model, model_choices
from faith.record_pipelines.prediction import model_querier

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "website/docs/configuration/model-configs.md"
QUERY = REPO / "website/docs/cli/query.md"
SELECTOR = REPO / "website/src/components/ModelConfig/TwoPaneConfig.js"

with patch.object(
    sys,
    "argv",
    ["faith", "query", "--model-paths", "fixture", "--datastore-location", "out/test"],
):
    from faith.cli.cli import _cli_parser, _cli_query


def snippet(title: str) -> str:
    """Read one uniquely titled code example from the model configuration page."""
    blocks = re.findall(
        rf'^```\w+ title="{re.escape(title)}"\n(.*?)^```',
        PAGE.read_text(),
        re.M | re.S,
    )
    if len(blocks) != 1:
        raise AssertionError(f"Expected one example titled {title!r}")
    return blocks[0]


class ModelConfigurationTests(unittest.TestCase):
    """Check published model settings with API calls replaced by test doubles."""

    # Independently reported unittest cases share the same isolated model setup.
    # pylint: disable=too-many-public-methods

    def setUp(self):
        # enterContext retains the context until unittest cleanup.
        # pylint: disable-next=consider-using-with
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(
            patch("socket.create_connection", side_effect=AssertionError("network"))
        )
        self.enterContext(
            patch("subprocess.Popen", side_effect=AssertionError("subprocess"))
        )
        sdk = ModuleType("openai")
        sdk.__dict__["OpenAI"] = type("OpenAI", (), {})
        # Import against a constructor double even if an SDK is installed.
        self.enterContext(patch.dict(sys.modules, {"openai": sdk}))
        # Import only after replacing the optional SDK with the fixture.
        # pylint: disable-next=import-outside-toplevel
        from faith.model.openai import OpenAIModel

        self.model_class = OpenAIModel
        self.sdk = self.enterContext(patch("faith.model.openai.OpenAI"))

    def resolve(self, command: str):
        """Resolve a documented command without querying a model."""
        argv = shlex.split(command.replace("\\\n", " "))
        self.assertEqual(argv[0], "faith")
        args = _cli_parser.parse_args(argv[1:])
        with patch("faith.cli.subcmd.query.run_experiment_queries") as run:
            _cli_query(args, Mock())
        return args, run.call_args.args[0]

    def example_configs(self):
        """Load every non-placeholder YAML configuration from the page."""
        blocks = re.findall(r"^```yaml[^\n]*\n(.*?)^```", PAGE.read_text(), re.M | re.S)
        configs = []
        for block in blocks:
            data = yaml.safe_load(block)["model"]
            if data["path"]:  # The schema intentionally has an empty placeholder.
                configs.append(ModelSpec.from_dict(data))
        self.assertGreaterEqual(len(configs), 4)
        return configs

    def test_yaml_examples_load_without_duplicate_or_reserved_kwargs(self):
        for spec in self.example_configs():
            with self.subTest(model=spec.path):
                self.assertFalse(
                    {
                        "temperature",
                        "top_p",
                        "max_completion_tokens",
                        "model",
                        "messages",
                        "tools",
                        "tool_choice",
                        "client",
                        "max_attempts",
                        "retry_sleep_secs",
                    }
                    & spec.generation.kwargs.keys()
                )
                self.assertFalse(
                    {
                        "num_gpus",
                        "context_len",
                        "name_or_path",
                        "tokenizer_name_or_path",
                        "reasoning_spec",
                        "seed",
                        "num_log_probs",
                    }
                    & spec.engine.kwargs.keys()
                )
                if spec.engine.engine_type != ModelEngine.VLLM:
                    self.assertIsNone(spec.reasoning)
                    self.assertEqual(spec.prompt_format, PromptFormatter.CHAT)
        local = ModelSpec.from_dict(
            yaml.safe_load(snippet("Local vLLM model"))["model"]
        )
        self.assertGreater(local.generation.max_completion_tokens, 0)
        self.assertLess(
            local.generation.max_completion_tokens, local.engine.context_length
        )

    def test_documented_override_command_uses_annotation_and_explicit_flags(self):
        args, (spec,) = self.resolve(snippet("Model config overrides"))
        original = load_model_config(choice_to_model("instruct/phi-4"))
        self.assertEqual(spec.engine.num_gpus, 2)
        self.assertEqual(spec.generation.temperature, 0.0)
        self.assertEqual(spec.generation.kwargs, {"top_k": 20})
        self.assertEqual(spec.engine.context_length, original.engine.context_length)
        self.assertEqual(
            spec.generation.max_completion_tokens,
            original.generation.max_completion_tokens,
        )
        self.assertEqual(args.sample_size, 25)
        self.assertEqual(args.num_trials, 1)
        self.assertEqual(str(args.datastore_location), "out/model-config")

    def test_omitted_flags_preserve_yaml_and_explicit_default_values_override(self):
        command = (
            "faith query --model-configs instruct/phi-4 --datastore-location out/test"
        )
        original = load_model_config(choice_to_model("instruct/phi-4"))
        _, (unchanged,) = self.resolve(command)
        self.assertEqual(unchanged, original)
        _, (changed,) = self.resolve(
            command + " --model-engine openai --model-context-len 3500"
            " --max-completion-tokens 500 --temperature 0.0"
            " --top-p 0.8 --prompt-format base"
        )
        self.assertEqual(changed.engine.engine_type, ModelEngine.VLLM)
        self.assertEqual(changed.engine.context_length, 3500)
        self.assertEqual(
            changed.generation,
            GenParams(temperature=0.0, top_p=0.8, max_completion_tokens=500),
        )
        self.assertEqual(changed.prompt_format, PromptFormatter.BASE)

    def test_yaml_include_is_recursive_but_cli_kwarg_merge_is_shallow(self):
        base = self.root / "base.yaml"
        base.write_text(
            yaml.safe_dump(
                {
                    "engine_type": "openai",
                    "kwargs": {
                        "api_num_threads": 3,
                        "nested": {"keep": 1, "replace": 2},
                    },
                }
            )
        )
        config = self.root / "model.yaml"
        config.write_text(
            "model:\n  path: fixture/model\n  prompt_format: chat\n"
            '  engine: !from\n    "base.yaml":\n      kwargs:\n'
            "        nested:\n          replace: 3\n"
            "  generation:\n    kwargs:\n      extra_body:\n        keep: 1\n        replace: 2\n"
        )
        inherited = load_model_config(config)
        self.assertEqual(inherited.engine.kwargs["nested"], {"keep": 1, "replace": 3})
        _, (overridden,) = self.resolve(
            f"faith query --model-configs {shlex.quote(str(config))} --datastore-location out/test "
            "--engine-kwargs 'nested={\"replace\": 4}' --generation-kwargs 'extra_body={\"replace\": 4}'"
        )
        self.assertEqual(
            overridden.engine.kwargs, {"api_num_threads": 3, "nested": {"replace": 4}}
        )
        self.assertEqual(overridden.generation.kwargs, {"extra_body": {"replace": 4}})

    def test_every_query_kwarg_fragment_passes_cli_literal_parsing(self):
        fragments = re.findall(
            r"^--(?:engine|generation)-kwargs .+$", QUERY.read_text(), re.M
        )
        self.assertGreaterEqual(len(fragments), 5)
        for fragment in fragments:
            with self.subTest(fragment=fragment):
                _, (spec,) = self.resolve(
                    "faith query --model-paths fixture --model-engine vllm --prompt-format chat --datastore-location out/test "
                    + fragment
                )
                if "dtype=" in fragment:
                    self.assertEqual(spec.engine.kwargs["dtype"], "bfloat16")

    def test_selector_command_resolves_every_packaged_config(self):
        matches = re.findall(r"\{`(faith query.*?)`\}", SELECTOR.read_text(), re.S)
        self.assertEqual(len(matches), 1)
        choices = model_choices()
        self.assertTrue(choices)
        for choice in choices:
            with self.subTest(choice=choice):
                command = (
                    matches[0].replace("${model.id}", choice).replace(r"\\\n", " ")
                )
                args, (spec,) = self.resolve(command)
                self.assertEqual(spec, load_model_config(choice_to_model(choice)))
                self.assertEqual(args.benchmarks, ["ctibench-mcqa"])
                self.assertEqual(args.sample_size, 25)
                self.assertEqual(args.num_trials, 1)
                self.assertEqual(str(args.datastore_location), "out/model-config")

    def test_openai_option_table_matches_wrapper_and_unknown_options_are_ignored(self):
        section = (
            QUERY.read_text()
            .split('<TabItem value="openai"', 1)[1]
            .split("</TabItem>", 1)[0]
        )
        documented = set(re.findall(r"^\| `([a-z_]+)` \|", section, re.M))
        constructor = inspect.signature(self.model_class).parameters
        supported = set(constructor) - {
            "name_or_path",
            "num_log_probs",
            "reasoning_spec",
            "_kwargs",
        }
        self.assertEqual(documented, supported)
        model = self.model_class(
            "fixture",
            api_key="fixture",
            base_url="http://localhost:8100/v1",
            timeout=5,
            max_retries=7,
        )
        self.sdk.assert_called_once_with(
            api_key="fixture", base_url="http://localhost:8100/v1", max_retries=0
        )
        self.assertEqual(model.api_max_attempts, 10)
        self.assertEqual(model.api_retry_sleep_secs, 1.0)

    def test_probe_and_agentic_turn_receive_generation_settings_with_distinct_limits(
        self,
    ):
        generation = GenParams(
            temperature=0.3,
            top_p=0.8,
            max_completion_tokens=2000,
            kwargs={"stop": ["fixture-stop"]},
        )
        model = Mock(name_or_path="fixture", api_max_attempts=3, api_retry_sleep_secs=0)
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(tool_calls=["probe"]), finish_reason="stop"
                )
            ]
        )
        model.client.chat.completions.create.return_value = response
        _check_tool_calling(model, generation)
        probe = model.client.chat.completions.create.call_args.kwargs
        self.assertEqual(probe["max_completion_tokens"], 1024)
        self.assertEqual(probe["tool_choice"], "required")
        model.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="fixture", tool_calls=[])
                )
            ]
        )
        loop = ReActLoop(
            model_api=model.client,
            model="fixture",
            environment=Mock(),
            tool_registry=ToolRegistry([]),
            generation_params=generation.as_request_params(),
        )
        result = loop.run(Task(id="fixture", prompt="fixture", ground_truth=None))
        self.assertIsNone(result.error)
        request = model.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["max_completion_tokens"], 2000)
        for key in ("temperature", "top_p", "stop"):
            self.assertEqual(probe[key], request[key])

    def test_negative_completion_limit_is_omitted_for_qa_and_agentic(self):
        generation = GenParams(max_completion_tokens=-1)
        self.assertNotIn("max_completion_tokens", generation.as_request_params())
        model = self.model_class("fixture", api_key="fixture")
        with patch.object(model, "_query_api", return_value=Mock()) as request:
            list(
                model.query(
                    [[{"role": "user", "content": "fixture"}]],
                    **generation.as_request_params(),
                    max_completion_tokens=-1,
                )
            )
        self.assertNotIn("max_completion_tokens", request.call_args.kwargs)

    def test_duplicate_generation_warning_matches_current_qa_and_agentic_behavior(self):
        generation = GenParams(kwargs={"temperature": 0.7})
        self.assertEqual(generation.as_request_params()["temperature"], 0.7)
        with self.assertRaisesRegex(TypeError, "multiple values.*temperature"):
            list(model_querier(Mock(), GenerationMode.CHAT_COMP, generation)([]))
        self.assertIn("do not repeat them in `generation.kwargs`", PAGE.read_text())

    def test_prompt_formats_and_api_reasoning_restriction(self):
        self.assertEqual(
            PromptFormatter.BASE.format("system", "question", None), "question\n"
        )
        self.assertEqual(
            PromptFormatter.CHAT.format("system", "question", None),
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
        )
        model = self.model_class("fixture", api_key="fixture")
        self.assertEqual(model.supported_formats, {PromptFormatter.CHAT})
        local = ModelSpec.from_dict(
            yaml.safe_load(snippet("Local vLLM model"))["model"]
        )
        with self.assertRaisesRegex(AssertionError, "Reasoning is not supported"):
            self.model_class("fixture", reasoning_spec=local.reasoning)

    def test_retry_attempt_count_and_probe_share_the_bounded_retry_policy(self):
        request = Mock(side_effect=ConnectionError("fixture failure"))
        with patch("faith._internal.functools.retriable.time.sleep"):
            with self.assertLogs(
                "faith._internal.functools.retriable", level="WARNING"
            ):
                with self.assertRaises(MaxAttemptsExceededError):
                    RetryFunctionWrapper(request, max_attempts=3, retry_sleep_secs=0)()
        self.assertEqual(request.call_count, 3)
        model = Mock(name_or_path="fixture", api_max_attempts=3, api_retry_sleep_secs=0)
        model.client.chat.completions.create.side_effect = ConnectionError(
            "fixture failure"
        )
        with self.assertLogs("faith._internal.functools.retriable", level="WARNING"):
            with self.assertRaisesRegex(ValueError, "probing.*failed"):
                _check_tool_calling(model, GenParams())
        self.assertEqual(model.client.chat.completions.create.call_count, 3)
        model.client.chat.completions.create.reset_mock()
        model.client.chat.completions.create.side_effect = RuntimeError(
            "fixture failure"
        )
        with self.assertRaisesRegex(ValueError, "probing.*failed"):
            _check_tool_calling(model, GenParams())
        model.client.chat.completions.create.assert_called_once()
        self.assertIn("SDK retries are disabled", PAGE.read_text())
        self.assertNotIn("not to the initial tool-calling probe", PAGE.read_text())

    def test_agentic_reserved_fields_and_streaming_are_rejected_before_probe(self):
        for field in (
            "model",
            "messages",
            "tools",
            "tool_choice",
            "client",
            "max_attempts",
            "retry_sleep_secs",
            "stream",
        ):
            for nested in (False, True):
                with self.subTest(field=field, extra_body=nested):
                    params: dict[str, Any] = {field: True}
                    if nested:
                        params = {"extra_body": params}
                    with self.assertRaises(ValueError):
                        validate_agentic_generation_params(params)
                    model = Mock()
                    with self.assertRaises(ValueError):
                        _check_tool_calling(model, GenParams(kwargs=params))
                    model.client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
