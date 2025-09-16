# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Provides the VLLM model backend for executing inference with VLLM supported models."""
import contextlib
import gc
import logging
from functools import partial
from typing import Any, Iterable, cast

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.outputs import RequestOutput

from faith.benchmark.formatting.prompt import PromptFormatter
from faith.model.base import (
    BaseModel,
    ChatResponse,
    GenerationError,
    PromptList,
    TokenPred,
    _is_message_list,
    _is_string_list,
)

logger = logging.getLogger(__name__)


class _VLLMBackend(BaseModel):
    """A base class for VLLM model backends that provides common functionality."""

    def __init__(
        self,
        name_or_path: str,
        tokenizer_name_or_path: str,
        num_gpus: int = 8,
        seed: int = 54748,
        context_len: int = 400,
        num_log_probs: int | None = None,
        **vllm_kwargs: Any,
    ):
        """Initialize the VLLM client for the model with the given parameters."""
        super().__init__(name_or_path)

        self._num_log_probs = num_log_probs
        if self._num_log_probs is not None and self._num_log_probs < 0:
            self._num_log_probs = self.tokenizer.vocab_size

        if self._num_log_probs is not None:
            vllm_kwargs["max_logprobs"] = self._num_log_probs
        self._model = LLM(
            model=self.name_or_path,
            tokenizer=tokenizer_name_or_path,
            tensor_parallel_size=num_gpus,
            max_model_len=context_len,
            seed=seed,
            **vllm_kwargs,
        )
        self._tokenizer = self._model.get_tokenizer()

    def __del__(self) -> None:
        """Clean up the model and distributed environment used by VLLM."""
        # Clean up the model and distributed environment from
        # https://github.com/vllm-project/vllm/issues/1908
        destroy_model_parallel()
        destroy_distributed_environment()

        # Only delete the model if it was correctly initialized; if there was an
        # error during initialization, the model may not exist, but __del__ is still
        # called on exit.
        if hasattr(self, "_model"):
            if hasattr(self._model, "llm_engine"):
                if hasattr(self._model.llm_engine, "model_executor"):
                    if hasattr(self._model.llm_engine.model_executor, "driver_worker"):
                        del self._model.llm_engine.model_executor.driver_worker
                    del self._model.llm_engine.model_executor
            del self._model
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Model and distributed environment cleaned up.")

    def _gen_outputs(
        self,
        inputs: PromptList,
        continue_final_message: bool,
        **gen_params: Any,
    ) -> list[RequestOutput]:
        """Generate outputs from the model for each of the input prompts."""
        if _is_message_list(inputs):
            assert (
                PromptFormatter.CHAT in self.supported_formats
            ), f"Chat format is not supported for the model {self.name_or_path}. Please use a different format."
            return self._model.chat(
                inputs,  # type: ignore[arg-type]
                sampling_params=SamplingParams(**gen_params),
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
                use_tqdm=partial(tqdm, leave=False),
            )
        elif _is_string_list(inputs):
            return self._model.generate(
                inputs,  # type: ignore[arg-type]
                sampling_params=SamplingParams(**gen_params),
                use_tqdm=partial(tqdm, leave=False),
            )
        else:
            raise ValueError(
                f"Unsupported input format. Expected a list of strings or messages, but got {type(inputs)}."
            )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the tokenizer used by the model."""
        return cast(PreTrainedTokenizerBase, self._tokenizer)


class VLLMModel(_VLLMBackend):
    """A model wrapper for the VLLM backend that supports various generation modes."""

    def __init__(
        self,
        name_or_path: str,
        tokenizer_name_or_path: str | None = None,
        num_gpus: int = 1,
        seed: int = 54748,
        context_len: int = 400,
        num_log_probs: int | None = None,
        **vllm_kwargs: Any,
    ):
        """Initialize the VLLM client for the model with the given parameters."""
        super().__init__(
            name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path or name_or_path,
            num_gpus=num_gpus,
            seed=seed,
            context_len=context_len,
            num_log_probs=num_log_probs,
            **vllm_kwargs,
        )

    @property
    def supported_formats(self) -> set[PromptFormatter]:
        """Return the supported input formats for the model."""
        formats = set([PromptFormatter.BASE])
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            formats.add(PromptFormatter.CHAT)
        return formats

    def logits(
        self,
        inputs: PromptList,
        max_answer_tokens: int = 1,
        **gen_params: Any,
    ) -> Iterable[list[list[TokenPred]] | GenerationError]:
        """Map each input in `inputs` to the model's next-token logits for it."""
        gen_params["max_tokens"] = max_answer_tokens
        gen_params["logprobs"] = self._num_log_probs
        return (
            [
                [
                    TokenPred(
                        token=self.tokenizer.decode(tokid, skip_special_tokens=True),
                        token_id=tokid,
                        logprob=lp.logprob,
                        rank=lp.rank,
                    )
                    for tokid, lp in per_token_info.items()
                ]
                for per_token_info in (output.outputs[0].logprobs or [])
            ]
            for output in self._gen_outputs(inputs, True, **gen_params)
        )

    def next_token(
        self,
        inputs: PromptList,
        verbose_resps: bool = False,
        **gen_params: Any,
    ) -> Iterable[ChatResponse | GenerationError]:
        """Map each input in `inputs` to the model's next generate token for it."""
        gen_params["max_tokens"] = 1
        return (
            ChatResponse(
                prompt_token_ids=output.prompt_token_ids if verbose_resps else None,
                num_prompt_tokens=len(output.prompt_token_ids or []),
                prompt_text=(
                    self.tokenizer.decode(
                        output.prompt_token_ids, skip_special_tokens=False
                    )
                    if verbose_resps
                    else None
                ),
                output_token_ids=output.outputs[0].token_ids if verbose_resps else None,
                num_output_tokens=len(output.outputs[0].token_ids),
                output_text=output.outputs[0].text,
                max_token_halt=output.outputs[0].finish_reason == "length",
            )
            for output in self._gen_outputs(inputs, True, **gen_params)
        )

    def query(
        self,
        inputs: PromptList,
        verbose_resps: bool = False,
        max_completion_tokens: int = 500,
        **gen_params: Any,
    ) -> Iterable[ChatResponse | GenerationError]:
        """Map each input in `inputs` to the model's generated response for it."""
        if max_completion_tokens > 0:
            gen_params["max_tokens"] = max_completion_tokens
        return (
            ChatResponse(
                prompt_token_ids=output.prompt_token_ids if verbose_resps else None,
                num_prompt_tokens=len(output.prompt_token_ids or []),
                prompt_text=(
                    self.tokenizer.decode(
                        output.prompt_token_ids, skip_special_tokens=False
                    )
                    if verbose_resps
                    else None
                ),
                output_token_ids=output.outputs[0].token_ids if verbose_resps else None,
                num_output_tokens=len(output.outputs[0].token_ids),
                output_text=output.outputs[0].text,
                max_token_halt=output.outputs[0].finish_reason == "length",
            )
            for output in self._gen_outputs(inputs, False, **gen_params)
        )
