# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Base TaskLoader abstraction for agentic evaluation."""

from abc import ABC, abstractmethod
from pathlib import Path

from faith._types.json_types import JsonValue
from faith.agentic.core.task import Task


class TaskLoader(ABC):
    """Base class for loading benchmark tasks.

    TaskLoaders discover and parse tasks from benchmark-specific formats.
    Each benchmark needs a custom loader to interpret its metadata.
    """

    def __init__(self, config: dict[str, JsonValue], working_dir: str):
        """Initialize loader with configuration.

        Args:
            config: Loader-specific configuration from benchmark.yaml
            working_dir: The sandbox directory the agent's commands will run in.
                A required argument rather than a config key so a loader whose
                prompt names the directory cannot silently disagree with the
                environment the agent runs in.
        """
        self._config = config
        self._working_dir = working_dir

    @abstractmethod
    def load_tasks(self, benchmark_path: Path) -> list[Task]:
        """Load all tasks from benchmark directory.

        Args:
            benchmark_path: The benchmark's definition directory (the one holding
                benchmark.yaml), which may also hold its data

        Returns:
            List of Task objects with id, prompt, ground_truth, metadata, etc.

        Raises:
            FileNotFoundError: If benchmark data is missing
            ValueError: If task metadata is invalid
        """
