# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Task dataclass for agentic evaluation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from faith._types.json_types import JsonValue


@dataclass
class Task:
    """Represents a single task in an agentic benchmark.

    Attributes:
        ground_truth: Expected answer or ground truth data, as parsed from the
            benchmark's data (any JSON type)
        metadata: Additional task-specific metadata (e.g., difficulty, category)
        data_path: Path to the task's data payload (directory or file) copied
            into the environment, or None if the task has no associated data
        data_provider: Called during environment setup to produce `data_path`,
            for a benchmark whose payload is fetched rather than shipped. Set
            alongside `data_path`, not instead of it: `data_path` must name where
            the payload will land, so it is known before the fetch happens.
        environment_config: Per-task environment overrides (empty if none). The
            highest of the three layers: it outranks both the benchmark's
            environment section and a deployment's `--environment-config`, since
            only the loader knows what a particular task needs.
        data_identity: Stable, JSON-serializable identity for the data payload.
            Unlike `data_path`, this is suitable for reconciliation because it
            must not include machine-specific cache directories. If omitted,
            the formatter retains the legacy `data_path` fallback.
    """

    id: str
    prompt: str
    ground_truth: JsonValue
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    data_path: Path | None = None
    data_provider: Callable[[], None] | None = None
    environment_config: dict[str, JsonValue] = field(default_factory=dict)
    data_identity: JsonValue | None = None
