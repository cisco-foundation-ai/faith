# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Write-side mapping between agent tasks and log records."""

from faith._internal.algo.hash import dict_sha256
from faith._internal.iter.transform import Mapping
from faith._types.record.model import ModelRecord
from faith._types.record.prompt import PromptRecord
from faith._types.record.sample import Metadata, SampleRecord
from faith._types.record.stats import Labeling
from faith.agentic.core import Task


class TaskFormatter(Mapping[tuple[int, Task], SampleRecord]):
    """Transform that formats each indexed task into a bare log record.

    Records carry no model output, so they can be reconciled against a prior
    trial log before the expensive agent loop runs. AgentLoopRunner fills the
    trajectory; AgentTrajectoryLogGrader fills stats at eval time. PromptRecord
    fields with no agentic meaning are left None.
    """

    def __init__(self, version: str, system_prompt: str | None):
        self._version = version
        self._system_prompt = system_prompt

    def _map_fn(self, element: tuple[int, Task]) -> SampleRecord:
        """Map one `(index, task)` pair into a bare log record, before the agent runs.

        The index is used with the sample hash as the record's reconciliation key.
        """
        index, task = element
        label: Labeling | None = task.ground_truth
        data = PromptRecord(
            benchmark_sample_index=index,
            # Every input that defines the task, so changing the payload or the
            # per-task environment invalidates a cached trajectory rather than
            # silently reusing one produced under different conditions.
            benchmark_sample_hash=dict_sha256(
                {
                    "id": task.id,
                    "prompt": task.prompt,
                    "ground_truth": task.ground_truth,
                    "data_path": str(task.data_path) if task.data_path else None,
                    "environment_config": task.environment_config,
                }
            ),
            subject=None,
            system_prompt=self._system_prompt,
            instruction=None,
            question=task.prompt,
            choices=None,
            label=label,
            formatted_question=task.prompt,
            formatted_answer=None,
            question_prompt=task.prompt,
            ancillary_data=task.metadata or None,
        )
        return SampleRecord(
            metadata=Metadata(
                version=self._version, data_hash=dict_sha256(data.to_dict())
            ),
            data=data,
            model_data=ModelRecord(
                prompt=task.prompt,
                answer_symbol_ids={},
                agent_trajectory=None,
            ),
            stats=None,
        )
