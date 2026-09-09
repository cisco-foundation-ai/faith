# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Write-side mapping from an agent run's result into a log record."""

from faith._internal.algo.hash import dict_sha256
from faith._types.record.agent import AgentResult, StoppedReason
from faith._types.record.model import ModelRecord
from faith._types.record.model_response import GenerationError
from faith._types.record.prompt import PromptRecord
from faith._types.record.sample import Metadata, SampleRecord
from faith._types.record.stats import Labeling
from faith.agentic.core.task import Task


def agent_result_to_sample_record(
    task: Task,
    result: AgentResult,
    *,
    index: int,
    version: str,
) -> SampleRecord:
    """Map one agent run into a stale log record for later grading.

    The record carries no scores; AgentTrajectoryLogGrader fills stats at eval
    time. PromptRecord fields with no agentic meaning are left None.

    Args:
        task: The task that was run; supplies the prompt and ground truth.
        result: The agent run's outcome, stored whole as the trajectory.
        index: The task's index within the benchmark, used with the sample
            hash as the record's reconciliation key.
        version: The benchmark version stamped into the record metadata.
    """
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
        system_prompt=None,
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
        metadata=Metadata(version=version, data_hash=dict_sha256(data.to_dict())),
        data=data,
        model_data=ModelRecord(
            prompt=task.prompt,
            answer_symbol_ids={},
            agent_trajectory=result,
            # A harness/environment failure is recorded the way QA records a
            # generation failure, so the shared aggregator excludes it from the
            # accuracy metrics instead of scoring it as a wrong answer. Only
            # ERROR qualifies: hitting a limit is a normal incomplete result.
            error=(
                GenerationError(title="Agent Run Error", details=result.error)
                if result.stopped_reason is StoppedReason.ERROR
                else None
            ),
        ),
        stats=None,
    )
