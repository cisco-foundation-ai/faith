# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Base AnswerExtractor abstraction for agentic evaluation."""

from abc import ABC, abstractmethod

from faith._types.json_types import JsonValue
from faith._types.record.stats import Labeling


class AnswerExtractor(ABC):
    """Turns an agent's submitted answer into a prediction the score_fns consume.

    Sees only the agent's output, never the ground truth: the agentic analog of
    QA's answer matcher.
    """

    def __init__(self, config: dict[str, JsonValue]):
        """Initialize the extractor with configuration.

        Args:
            config: Extractor-specific configuration from benchmark.yaml, not
                metric config.
        """
        self._config = config

    @abstractmethod
    def extract(self, tool_name: str | None, answer: JsonValue) -> Labeling | None:
        """Extract a scorable prediction from an agent's submitted answer.

        Args:
            tool_name: Which terminal tool ended the run, or None if the agent
                never submitted. Benchmarks with several terminal tools (submit
                vs abstain) may map the same answer differently by tool.
            answer: Agent's submitted answer, exactly as supplied to the terminal
                tool (any JSON type), or None if the agent never submitted.

        Returns:
            The prediction, or None if the answer does not extract (never
            submitted, or an unusable shape).
        """
