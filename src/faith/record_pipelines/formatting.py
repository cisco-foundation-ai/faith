# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from transformers import PreTrainedTokenizerBase

from faith._internal.algo.hash import dict_sha256
from faith._internal.iter.transform import Mapping
from faith._types.model.generation import GenerationMode
from faith._types.record.model import ModelRecord
from faith._types.record.prompt import PromptRecord
from faith._types.record.sample import Metadata, SampleRecord
from faith.benchmark.benchmark import Benchmark


class SampleFormatter(Mapping[PromptRecord, SampleRecord]):
    """Transform that converts PromptRecords into log records for querying the model."""

    def __init__(self, benchmark: Benchmark, tokenizer: PreTrainedTokenizerBase | None):
        self._bench_formatter = benchmark.formatter
        self._bench_version = benchmark.version

        # Create the lead-in string for the answer portion of the prompt.
        self._answer_leadin = None
        if benchmark.generation_mode in [
            GenerationMode.LOGITS,
            GenerationMode.NEXT_TOKEN,
        ]:
            assert (
                tokenizer is not None
            ), f"Model tokenizer required for {str(benchmark.generation_mode)}."
            self._answer_leadin = benchmark.answer_leadin(tokenizer)

        # Translate the answer symbols to the model's tokenizer's token IDs.
        self._answer_symbol_ids = {}
        if benchmark.generation_mode == GenerationMode.LOGITS:
            assert hasattr(
                benchmark, "answer_token_map"
            ), "Token map required for logits generation."
            assert (
                tokenizer is not None
            ), "Model tokenizer is required for logits generation."
            self._answer_symbol_ids = benchmark.answer_token_map(tokenizer)

    def _map_fn(self, element: PromptRecord) -> SampleRecord:
        """Map a PromptRecord into a log record containing the data and metadata for querying."""
        return SampleRecord(
            metadata=Metadata(
                version=self._bench_version,
                data_hash=dict_sha256(element.to_dict()),
            ),
            data=element,
            model_data=ModelRecord(
                prompt=self._bench_formatter.render_conversation(
                    element, self._answer_leadin
                ),
                answer_symbol_ids=self._answer_symbol_ids,
            ),
            stats=None,
        )
