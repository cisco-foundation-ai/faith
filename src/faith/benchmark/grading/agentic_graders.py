# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides the log grader for agentic benchmarks."""

from faith._types.config.patterns import AnswerFormat
from faith._types.config.scoring import OutputProcessingConfig
from faith._types.record.sample import SampleRecord
from faith._types.record.stats import Labeling, StatsRecord
from faith.agentic.core import AnswerExtractor
from faith.benchmark.grading.log_grader import LogGrader


class AgentTrajectoryLogGrader(LogGrader):
    """A log grader for agentic benchmarks that log an agent trajectory.

    Reads ``model_data.agent_trajectory`` as ChatCompletionLogGrader reads
    ``chat_comp``. It does not parse the answer itself: a benchmark-specific
    AnswerExtractor turns the submitted answer into a prediction, which the
    configured score_fns then grade against the record's label.
    """

    def __init__(
        self,
        output_processing_config: OutputProcessingConfig,
        extractor: AnswerExtractor,
    ):
        """Initialize the agent-trajectory log grader.

        Args:
            output_processing_config: Metric configuration; its score_fns are the
                metric functions applied to the extracted prediction and label.
            extractor: Benchmark-specific extractor turning the raw submitted
                answer into a prediction. Injected here because class resolution
                is not wired yet; a later PR constructs it from the benchmark
                config and injects it.
        """
        super().__init__(output_processing_config)
        self._extractor = extractor

    def _markup_entry(self, log_entry: SampleRecord) -> SampleRecord:
        """Markup a single log entry with the computed statistics / scores."""
        label: Labeling | None = log_entry.data.label

        agent_trajectory = log_entry.model_data.agent_trajectory
        answer = agent_trajectory.answer if agent_trajectory is not None else None
        answer_tool = (
            agent_trajectory.answer_tool if agent_trajectory is not None else None
        )
        prediction = self._extractor.extract(answer_tool, answer)

        # answer_format captures whether the agent produced a scorable answer:
        # nothing submitted is INVALID; a submitted answer that does not extract
        # for this benchmark is IMPROPER; a cleanly extracted answer is PROPER.
        # answer_tool, not answer, is what marks a submission: a terminal tool may
        # legitimately submit null (an abstain, or an explicit "no answer").
        if prediction is not None:
            answer_format = AnswerFormat.PROPER
        elif answer_tool is not None:
            answer_format = AnswerFormat.IMPROPER
        else:
            answer_format = AnswerFormat.INVALID

        log_entry.stats = StatsRecord(
            label=label,
            prediction=prediction,
            answer_format=answer_format,
            subject=log_entry.data.subject,
            scores=self._custom_scores(
                label,
                prediction,
                ancillary_data=log_entry.data.ancillary_data,
            ),
            error=error.title if (error := log_entry.model_data.error) else None,
            stopped_reason=(
                str(agent_trajectory.stopped_reason)
                if agent_trajectory is not None
                else None
            ),
        )
        return log_entry
