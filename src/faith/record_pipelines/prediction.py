# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Type

from faith._internal.iter.transform import IsoTransform
from faith._types.enums import CIEnum
from faith._types.model.generation import GenerationMode, GenParams
from faith._types.record.model_response import ChatResponse, GenerationError
from faith._types.record.sample import SampleRecord
from faith.model.base import BaseModel


class _PredictionTransform(IsoTransform[SampleRecord]):
    """Base class for prediction transforms that generate model outputs."""

    def __init__(self, model: BaseModel, gen_params: GenParams):
        """Initialize the prediction transform for a model."""
        super().__init__()
        self._model = model
        self._gen_params = gen_params


class _LogitsTransform(_PredictionTransform):
    """Transform for generating logits from a model."""

    def __call__(self, records: Iterable[SampleRecord]) -> Iterable[SampleRecord]:
        """Generate the next-token logits for each input in `records`."""
        inputs = list(records)
        logit_responses = self._model.logits(
            inputs=[example.model_data.prompt for example in inputs],
            temperature=self._gen_params.temperature,
            top_p=self._gen_params.top_p,
            **self._gen_params.kwargs,
        )
        for record, logit_response in zip(inputs, logit_responses):
            if isinstance(logit_response, list):
                record.model_data.logits = logit_response
            elif isinstance(logit_response, GenerationError):
                record.model_data.error = logit_response
            yield record


class _NextTokenTransform(_PredictionTransform):
    """Transform for generating next token predictions from a model."""

    def __call__(self, records: Iterable[SampleRecord]) -> Iterable[SampleRecord]:
        """Generate next token predictions for each input in `records`."""
        inputs = list(records)
        responses = self._model.next_token(
            inputs=[example.model_data.prompt for example in inputs],
            temperature=self._gen_params.temperature,
            top_p=self._gen_params.top_p,
            **self._gen_params.kwargs,
        )
        for record, response in zip(inputs, responses):
            if isinstance(response, ChatResponse):
                record.model_data.next_token = response
            elif isinstance(response, GenerationError):
                record.model_data.error = response
            yield record


class _GenerationTransform(_PredictionTransform):
    """Transform for generating chat completions from a model."""

    def __call__(self, records: Iterable[SampleRecord]) -> Iterable[SampleRecord]:
        """Generate chat completion responses for each input in `records`."""
        inputs = list(records)
        responses = self._model.query(
            inputs=[example.model_data.prompt for example in inputs],
            temperature=self._gen_params.temperature,
            max_completion_tokens=self._gen_params.max_completion_tokens,
            top_p=self._gen_params.top_p,
            **self._gen_params.kwargs,
        )
        for record, response in zip(inputs, responses):
            if isinstance(response, ChatResponse):
                record.model_data.chat_comp = response
            elif isinstance(response, GenerationError):
                record.model_data.error = response
            yield record


class _ModelMethod(CIEnum):
    """Enumeration of model methods for generating predictions.

    Each enum value corresponds to a specific generative method of the model,
    allowing for flexible handling of different prediction types.
    """

    LOGITS = (_LogitsTransform,)
    NEXT_TOKEN = (_NextTokenTransform,)
    GENERATION = (_GenerationTransform,)

    @property
    def transform_cls(self) -> Type[_PredictionTransform]:
        """Return the transform class for this model method."""
        return self.value[0]

    def create_transform(
        self, model: BaseModel, gen_params: GenParams
    ) -> IsoTransform[SampleRecord]:
        """Create an instance of the transform for the model and generation parameters."""
        return self.transform_cls(model, gen_params)


def model_querier(
    model: BaseModel,
    generation_mode: GenerationMode,
    gen_params: GenParams,
) -> IsoTransform[SampleRecord]:
    """Create a transform that queries the model according to the specified generation mode."""
    mode_map = {
        GenerationMode.LOGITS: _ModelMethod.LOGITS,
        GenerationMode.NEXT_TOKEN: _ModelMethod.NEXT_TOKEN,
        GenerationMode.CHAT_COMP: _ModelMethod.GENERATION,
    }
    return mode_map[generation_mode].create_transform(model, gen_params)
