# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""ReAct agent loop orchestration."""

import json
import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, TypedDict

from faith._internal.functools.retriable import (
    MaxAttemptsExceededError,
    RetryFunctionWrapper,
)
from faith._types.json_types import JsonValue
from faith._types.record.agent import (
    AgentResult,
    StoppedReason,
    ToolCallEntry,
    ToolResponseEntry,
)
from faith.agentic.core.environment import Environment
from faith.agentic.core.task import Task
from faith.agentic.core.tool import Tool
from faith.agentic.core.tool_registry import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
)

logger = logging.getLogger(__name__)

# Values of the "error" key in a tool_calls entry. Each marks a call that did
# NOT execute a tool, so it is not charged against max_tool_calls; an executed
# call carries "args" instead. The last is a well-formed call arriving after the
# budget is spent, rather than a malformed one.
INVALID_JSON_ERROR = "invalid_json"
INVALID_ARGS_ERROR = "invalid_args"
INVALID_SUBMIT_ERROR = "invalid_submit"
UNKNOWN_TOOL_ERROR = "unknown_tool"
BUDGET_EXHAUSTED_ERROR = "budget_exhausted"

# Tool responses are truncated to this many characters in the trajectory log.
MAX_LOGGED_RESPONSE_CHARS = 500

# A well-behaved tool caps its own output; this is the backstop that keeps a tool
# that doesn't from exceeding the model API's request limit and failing the task.
MAX_TOOL_RESULT_CHARS = 100_000


class _FunctionCall(TypedDict):
    """The function half of a model's tool call."""

    name: str
    arguments: str


class _ModelToolCall(TypedDict):
    """One tool call as normalized from the model response."""

    id: str
    type: str
    function: _FunctionCall


@dataclass
class _ToolCallOutcome:
    """Result of processing a single tool call, for the run loop to apply.

    Attributes:
        call_log: Entry to append to AgentResult.tool_calls.
        response_log: Entry to append to AgentResult.tool_responses.
        message: Tool-result message to append to the conversation.
        submitted: This was a terminal tool, so the run should end.
        answer: The submitted answer, raw as the model supplied it (the value at
            the tool's answer_field, or None for an identity-carried terminal
            tool). Meaningful only when submitted is True.
        answer_tool: Name of the terminal tool, recorded on the result so the
            extractor can grade by tool identity. Meaningful only when submitted
            is True.
    """

    call_log: ToolCallEntry
    response_log: ToolResponseEntry
    message: dict[str, Any]
    submitted: bool
    answer: JsonValue = None
    answer_tool: str | None = None


def validate_agent_limits(
    max_tool_calls: int,
    max_turns: int,
    max_consecutive_no_tool_calls: int,
    timeout_seconds: float | None,
) -> None:
    """Validate agent limits, shared with the yaml-facing AgenticLimitsConfig.

    Both validate: a yaml benchmark goes through AgenticLimitsConfig, a library
    caller constructs ReActLoop directly.

    Raises:
        ValueError: If any limit is out of range.
    """
    if max_tool_calls < 1 or max_turns < 1:
        raise ValueError(
            f"max_tool_calls ({max_tool_calls}) and max_turns "
            f"({max_turns}) must both be at least 1."
        )
    if max_turns < max_tool_calls:
        raise ValueError(
            f"max_turns ({max_turns}) must be >= max_tool_calls ({max_tool_calls})."
        )
    if max_consecutive_no_tool_calls < 1:
        raise ValueError(
            f"max_consecutive_no_tool_calls ({max_consecutive_no_tool_calls}) "
            f"must be at least 1."
        )
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError(
            f"timeout_seconds ({timeout_seconds}) must be positive; "
            f"omit it for no timeout."
        )


@dataclass(frozen=True)
class ReActConfig:
    """Immutable configuration for ReAct runs: the reusable knobs and clients.

    This is data, not behavior — a _Run reads it to execute one task. Kept
    separate from _Run (per-task mutable state) and AgentResult (the outcome).

    Attributes:
        model_api: Client for making LLM calls, exposing the OpenAI-compatible
            chat.completions.create(...) surface (real OpenAI/OpenRouter, or a
            vLLM server reached via base_url). Typed Any until this is wired to
            a real client; a structural Protocol is the eventual type.
        model: Model identifier passed as the `model=` argument on every
            chat.completions.create call (e.g. "meta-llama/Llama-3-70b").
            Required by the API to select which model serves the request.
        system_prompt: Optional system message seeded ahead of the task prompt,
            carrying the benchmark's standing instructions (role, method, effort).
            None sends no system message.
        max_tool_calls: Budget for calls that execute a tool, checked before each
            call. Calls that execute nothing (malformed arguments, unknown tool,
            a call refused for want of budget) are not charged, and neither is a
            terminal call: submitting is not exploration.
        max_turns: Hard cap on model turns, bounding unproductive turns that
            never spend the tool budget. Must be >= max_tool_calls.
        max_consecutive_no_tool_calls: How many prose-only turns in a row end the
            run as NO_TOOL_CALLS. Earlier ones are nudged instead, so one chatty
            turn does not discard the task. 1 restores fail-on-first behavior.
        timeout_seconds: Optional task timeout, checked BETWEEN turns, so an
            in-flight call is not cancelled and the run may overshoot it by one
            slow call. None means no timeout.
        api_max_attempts: Attempts per model call before the run fails, matching
            APIBasedModel's default. A transient API failure mid-run would
            otherwise discard the whole trajectory.
        api_retry_sleep_secs: Delay between those attempts.
        generation_params: Extra keyword arguments spread into every
            chat.completions.create call (e.g. temperature, top_p,
            max_completion_tokens). Kept as a plain dict so the loop stays
            decoupled from the model layer's GenParams. Empty means "use API
            defaults".
    """

    model_api: Any
    model: str
    environment: Environment
    tool_registry: ToolRegistry
    system_prompt: str | None = None
    max_tool_calls: int = 50
    max_turns: int = 100
    max_consecutive_no_tool_calls: int = 3
    timeout_seconds: float | None = None
    api_max_attempts: int = 10
    api_retry_sleep_secs: float = 1.0
    generation_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.api_max_attempts < 1:
            raise ValueError(
                f"api_max_attempts ({self.api_max_attempts}) must be at least 1."
            )
        if self.api_retry_sleep_secs < 0:
            raise ValueError(
                f"api_retry_sleep_secs ({self.api_retry_sleep_secs}) must be "
                f"non-negative."
            )
        validate_agent_limits(
            self.max_tool_calls,
            self.max_turns,
            self.max_consecutive_no_tool_calls,
            self.timeout_seconds,
        )
        self._validate_capabilities()
        # Evaluate now: a tool whose schema properties raise should fail here,
        # not on the first model call.
        _ = self.tool_schemas

    def _validate_capabilities(self) -> None:
        """Reject a registered tool this environment cannot serve."""
        env_name = type(self.environment).__name__
        for name in self.tool_registry.list_tools():
            tool = self.tool_registry.get_tool(name)
            if missing := tool.missing_capabilities(self.environment):
                required = ", ".join(cap.__name__ for cap in missing)
                raise ValueError(
                    f"Environment '{env_name}' does not support tool '{name}' "
                    f"(requires {required}). Registered tools: "
                    f"{', '.join(self.tool_registry.list_tools())}."
                )

    @cached_property
    def tool_schemas(self) -> list[dict[str, Any]]:
        """Function schemas for the model, derived once from the tool registry."""
        return self.tool_registry.get_function_schemas()

    @cached_property
    def terminal_tool_schemas(self) -> list[dict[str, Any]]:
        """The terminal subset of tool_schemas, offered on the final turn alone."""
        return [
            schema
            for schema in self.tool_schemas
            if self.tool_registry.get_tool(schema["function"]["name"]).is_terminal
        ]


class ReActLoop:
    """ReAct agent loop orchestrator.

    Manages the agent's reasoning and action loop:
    1. Sends task prompt to model
    2. Model responds with reasoning and/or tool calls
    3. Executes tool calls in environment
    4. Sends tool results back to model
    5. Repeats until submission or termination condition

    Termination conditions (see StoppedReason):
    - Agent calls a terminal tool, e.g. submit (SUBMITTED)
    - Model returns no tool calls (NO_TOOL_CALLS)
    - Tool-call budget reached (MAX_CALLS)
    - Turn cap reached (MAX_TURNS)
    - Wall-clock timeout reached (TIMEOUT)
    - Harness/API/environment failure (ERROR)

    This is the reusable, configured entry point; each run() delegates to a
    fresh _Run holding that task's mutable state.
    """

    def __init__(
        self,
        model_api: Any,  # Client exposing the OpenAI chat.completions.create shape
        model: str,
        environment: Environment,
        tool_registry: ToolRegistry,
        *,
        system_prompt: str | None = None,
        max_tool_calls: int = 50,
        max_turns: int = 100,
        max_consecutive_no_tool_calls: int = 3,
        timeout_seconds: float | None = None,
        api_max_attempts: int = 10,
        api_retry_sleep_secs: float = 1.0,
        generation_params: dict[str, Any] | None = None,
    ):
        """Initialize ReAct loop. See ReActConfig for argument semantics."""
        self._config = ReActConfig(
            model_api=model_api,
            model=model,
            environment=environment,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            max_tool_calls=max_tool_calls,
            max_turns=max_turns,
            max_consecutive_no_tool_calls=max_consecutive_no_tool_calls,
            timeout_seconds=timeout_seconds,
            api_max_attempts=api_max_attempts,
            api_retry_sleep_secs=api_retry_sleep_secs,
            generation_params=generation_params or {},
        )

    def run(self, task: Task) -> AgentResult:
        """Run agent on a task.

        Args:
            task: Task to run agent on

        Returns:
            AgentResult with answer, messages, tool calls, etc.
        """
        return _Run(self._config, task).execute()


class _Run:
    """A single execution of a ReActLoop over one task.

    Owns the run's mutable state as instance attributes; methods mutate self
    rather than threading a state object around. One _Run is created per run()
    call and never shared or reused, which is what makes mutating self safe.
    """

    def __init__(self, config: ReActConfig, task: Task):
        self._config = config
        self._task = task
        self._start_time = time.time()

        self._messages: list[dict[str, Any]] = []
        self._tool_calls_log: list[ToolCallEntry] = []
        self._tool_responses_log: list[ToolResponseEntry] = []
        self._executed_tool_calls = 0
        self._turn_count = 0
        self._consecutive_no_tool_calls = 0
        self._elapsed_seconds = 0.0
        self._answer: JsonValue = None
        self._answer_tool: str | None = None
        # Set once when the budget runs out, which is what bounds the extra turn
        # to exactly one.
        self._final_turn_offered = False
        # None means "still running"; the loop and _run_turn set it once.
        self._stopped_reason: StoppedReason | None = None
        self._error: str | None = None

    def execute(self) -> AgentResult:
        """Drive the run start-to-finish and build its AgentResult."""
        try:
            try:
                self._config.environment.setup(self._task)
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Setup runs inside the outer try so teardown still fires: a
                # setup that fails partway (e.g. container started, later step
                # raised) must be cleaned up, not leaked.
                self._stopped_reason = StoppedReason.ERROR
                self._error = f"Environment setup failed: {str(e)}"
            else:
                # Seeded only after setup succeeds, so a setup failure yields
                # empty buffers (the run never entered the loop).
                self._messages = self._initial_messages()

                while self._stopped_reason is None:
                    self._run_one_iteration()

        except Exception as e:  # pylint: disable=broad-exception-caught
            self._stopped_reason = StoppedReason.ERROR
            self._error = f"Agent loop error: {str(e)}"

        finally:
            # Stop the clock first: cleanup time is not the task's time.
            self._elapsed_seconds = time.time() - self._start_time
            # Teardown is not part of the task outcome, so a failure here must
            # not touch stopped_reason/error.
            try:
                self._config.environment.teardown()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning("Environment teardown failed: %s", e)

        return self._build_result()

    def _run_one_iteration(self) -> None:
        """Check the pre-turn limits, then run a turn if none of them stopped us.

        Limits are checked before each turn; outcomes (submitted / no tool calls)
        are set by the turn itself. Timeout is checked first so a run that overran
        the clock reports TIMEOUT rather than whichever budget it happened to
        exhaust on the same turn.
        """
        if self._timed_out():
            self._stopped_reason = StoppedReason.TIMEOUT
        elif self._executed_tool_calls >= self._config.max_tool_calls:
            # An agent that spent its budget exploring still has an answer worth
            # asking for, so the budget buys one final turn to submit it. The flag
            # bounds that to exactly one, and a registry with nothing to submit
            # through gets no turn at all.
            if self._final_turn_offered or not self._config.terminal_tool_schemas:
                self._stopped_reason = StoppedReason.MAX_CALLS
            else:
                self._final_turn_offered = True
                self._messages.append(self._budget_exhausted_message())
                self._take_turn(terminal_only=True)
        elif self._turn_count >= self._config.max_turns:
            self._stopped_reason = StoppedReason.MAX_TURNS
        else:
            self._take_turn()

    def _take_turn(self, terminal_only: bool = False) -> None:
        """Run one turn and count it.

        Args:
            terminal_only: Offer only the terminal tools, for the final turn.
        """
        self._run_turn(terminal_only)
        self._turn_count += 1

    def _timed_out(self) -> bool:
        """Whether the between-turns wall-clock timeout has been reached."""
        if self._config.timeout_seconds is None:
            return False
        return (time.time() - self._start_time) > self._config.timeout_seconds

    def _initial_messages(self) -> list[dict[str, Any]]:
        """The opening conversation: the benchmark's system prompt, then the task."""
        messages: list[dict[str, Any]] = []
        if self._config.system_prompt:
            messages.append({"role": "system", "content": self._config.system_prompt})
        messages.append({"role": "user", "content": self._task.prompt})
        return messages

    def _budget_exhausted_message(self) -> dict[str, Any]:
        """Prompt for the final turn the exhausted tool-call budget buys.

        Names the terminal tools specifically: the point of the turn is to get an
        answer, and every other tool is now refused.
        """
        names = ", ".join(
            name
            for name in self._config.tool_registry.list_tools()
            if self._config.tool_registry.get_tool(name).is_terminal
        )
        return {
            "role": "user",
            "content": (
                f"You have used all {self._config.max_tool_calls} of your tool "
                f"calls. Submit your answer now using: {names}."
            ),
        }

    def _nudge_message(self) -> dict[str, Any]:
        """Prompt for a turn that produced prose, naming the callable tools.

        Read from the registry rather than hardcoded: which tools exist, and
        which of them ends the run, is the benchmark's choice.
        """
        names = ", ".join(self._config.tool_registry.list_tools())
        return {
            "role": "user",
            "content": f"Please respond with a tool call. Available tools: {names}.",
        }

    def _run_turn(self, terminal_only: bool = False) -> None:
        """Run one model turn: call the model, then execute its tool calls.

        Mutates self, setting stopped_reason if the turn ends the run (the model
        submitted, or went max_consecutive_no_tool_calls turns without calling a
        tool); otherwise leaves it None to continue. The tool-call budget is
        charged and enforced per call by _execute_tool_call, so a turn's batch
        cannot overshoot it.

        Args:
            terminal_only: Offer only the terminal tools, for the final turn.
        """
        response = self._call_model(terminal_only)
        self._messages.append(response)

        tool_calls: list[_ModelToolCall] = response.get("tool_calls") or []
        if not tool_calls:
            # A single chatty turn is recoverable, so nudge and let the loop
            # continue; only a persistent refusal to call anything ends the run.
            self._consecutive_no_tool_calls += 1
            if (
                self._consecutive_no_tool_calls
                >= self._config.max_consecutive_no_tool_calls
            ):
                self._stopped_reason = StoppedReason.NO_TOOL_CALLS
            else:
                self._messages.append(self._nudge_message())
            return

        self._consecutive_no_tool_calls = 0

        # Every call gets a result message up to the terminal one, after which
        # the run is over and the remaining calls are left unanswered. The budget
        # is charged inside _execute_tool_call, which needs the live count to
        # report what remains.
        for tool_call in tool_calls:
            outcome = self._execute_tool_call(tool_call)
            self._tool_calls_log.append(outcome.call_log)
            self._tool_responses_log.append(outcome.response_log)
            self._messages.append(outcome.message)

            # A terminal tool ends the run: any remaining calls in this turn are
            # skipped.
            if outcome.submitted:
                self._answer = outcome.answer
                self._answer_tool = outcome.answer_tool
                self._stopped_reason = StoppedReason.SUBMITTED
                return

    def _build_result(self) -> AgentResult:
        """Build the AgentResult from the run's final state.

        Covers both a completed run and a setup failure that never entered the
        loop (empty buffers, stopped_reason ERROR).
        """
        assert self._stopped_reason is not None, "run ended without a stop reason"
        return AgentResult(
            task_id=self._task.id,
            answer=self._answer,
            answer_tool=self._answer_tool,
            messages=self._messages,
            tool_calls=self._tool_calls_log,
            tool_responses=self._tool_responses_log,
            num_tool_calls=len(self._tool_calls_log),
            execution_time_seconds=self._elapsed_seconds,
            stopped_reason=self._stopped_reason,
            final_turn_offered=self._final_turn_offered,
            error=self._error,
        )

    def _recoverable_error(
        self,
        tool_call: _ModelToolCall,
        call_timestamp: float,
        error: str,
        message: str,
    ) -> _ToolCallOutcome:
        """Build the outcome for a call fed back to the model without executing.

        Used for malformed arguments, an unknown tool name, and a call refused for
        want of budget — all of which the model can retry from. Nothing ran, so
        the call isn't charged against the budget or treated as a submission.
        """
        return _ToolCallOutcome(
            call_log={
                "tool_name": tool_call["function"]["name"],
                "raw_arguments": tool_call["function"]["arguments"],
                "error": error,
                "timestamp": call_timestamp,
            },
            response_log={
                "tool_name": tool_call["function"]["name"],
                "response_truncated": message[:MAX_LOGGED_RESPONSE_CHARS],
                "response_length": len(message),
                "timestamp": call_timestamp,
            },
            message={
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": message,
            },
            submitted=False,
        )

    def _parse_tool_args(
        self, tool_call: _ModelToolCall, call_timestamp: float
    ) -> dict[str, JsonValue] | _ToolCallOutcome:
        """A call's arguments as a dict, or the outcome that rejects them."""
        # A zero-argument call is served as null or "" by some OpenAI-compatible
        # servers; both mean "no arguments", not malformed.
        raw_args = tool_call["function"]["arguments"] or "{}"
        try:
            tool_args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return self._recoverable_error(
                tool_call,
                call_timestamp,
                INVALID_JSON_ERROR,
                f"Error: Model returned invalid JSON for tool arguments: {str(e)}",
            )

        if not isinstance(tool_args, dict):
            return self._recoverable_error(
                tool_call,
                call_timestamp,
                INVALID_ARGS_ERROR,
                f"Error: Tool arguments must be a JSON object; got: {raw_args}",
            )
        return tool_args

    def _observation(self, result: str, *, charge: bool) -> str:
        """What the model sees for an executed call: its result, bounded and costed.

        Charging happens here rather than in the caller because the remaining
        count has to include this call. A terminal call is not charged — see
        ReActConfig.max_tool_calls — and gets no remaining-count suffix, since the
        run ends before another call can use it.
        """
        if len(result) > MAX_TOOL_RESULT_CHARS:
            result = (
                result[:MAX_TOOL_RESULT_CHARS]
                + f"\n\n[TRUNCATED BY HARNESS — tool returned {len(result)} chars, "
                f"limit {MAX_TOOL_RESULT_CHARS}]"
            )
        if not charge:
            return result
        self._executed_tool_calls += 1
        remaining = self._config.max_tool_calls - self._executed_tool_calls
        return f"{result}\n[{remaining} tool-calls remaining]"

    def _terminal_tool(self, tool_name: str) -> Tool | None:
        """The registered terminal tool of this name, or None.

        Returns None for a non-terminal tool and for an unknown name (which is
        not terminal and falls through to execute_tool's not-found handling),
        so the caller can treat "is this call a submission?" as a simple
        None-check.
        """
        try:
            tool = self._config.tool_registry.get_tool(tool_name)
        except ToolNotFoundError:
            return None
        return tool if tool.is_terminal else None

    def _execute_tool_call(self, tool_call: _ModelToolCall) -> _ToolCallOutcome:
        """Process one tool call from a model turn.

        Parses the arguments and runs the tool, or records a malformed-JSON
        attempt without executing anything. Builds the log entries and the
        tool-result message for execute() to apply; has no side effects.

        Args:
            tool_call: One entry from a response's tool_calls list.

        Returns:
            A _ToolCallOutcome describing what the run should record.
        """
        tool_name = tool_call["function"]["name"]
        tool_call_id = tool_call["id"]
        call_timestamp = time.time() - self._start_time
        terminal_tool = self._terminal_tool(tool_name)

        # Refused rather than executed once the budget is gone, so the batch a
        # model emits cannot overshoot it. Terminal tools are exempt: submitting
        # is not exploration, and refusing it would discard the answer.
        if (
            terminal_tool is None
            and self._executed_tool_calls >= self._config.max_tool_calls
        ):
            return self._recoverable_error(
                tool_call,
                call_timestamp,
                BUDGET_EXHAUSTED_ERROR,
                f"ERROR: Tool call budget exhausted "
                f"({self._config.max_tool_calls}/{self._config.max_tool_calls}). "
                f"Please submit your answer now.",
            )

        parsed = self._parse_tool_args(tool_call, call_timestamp)
        if isinstance(parsed, _ToolCallOutcome):
            return parsed
        tool_args = parsed

        # Read before execution, so a terminal tool that then fails still ends the
        # run as SUBMITTED with the answer the model supplied.
        answer: JsonValue = None
        answer_tool: str | None = None
        if terminal_tool is not None:
            answer_tool = tool_name
            # No answer_field means the call itself is the whole signal (e.g.
            # abstain); a declared field that is missing is a malformed
            # submission, fed back so the model can retry it.
            if (answer_field := terminal_tool.answer_field) is not None:
                if answer_field not in tool_args:
                    return self._recoverable_error(
                        tool_call,
                        call_timestamp,
                        INVALID_SUBMIT_ERROR,
                        f"Error: {tool_name} requires a JSON object with a "
                        f"'{answer_field}' field, e.g. "
                        f'{{"{answer_field}": ...}}; '
                        f'got: {tool_call["function"]["arguments"]}',
                    )
                answer = tool_args[answer_field]

        try:
            result = self._config.tool_registry.execute_tool(
                tool_name, tool_args, self._config.environment
            )
        except ToolNotFoundError as e:
            # Unknown tool name: nothing executed, so feed it back like malformed
            # JSON and don't count it against the budget (max_turns bounds these).
            # Narrow on purpose: a ValueError raised by a tool that DID run is a
            # ToolExecutionError, and must not be reported as a bad tool name.
            return self._recoverable_error(
                tool_call,
                call_timestamp,
                UNKNOWN_TOOL_ERROR,
                f"Unknown tool: {str(e)}",
            )
        except ToolExecutionError as e:
            # The tool executed and raised: counts as executed, fed back so the
            # model can adjust rather than ending the run.
            result = f"Error executing tool: {str(e)}"

        # A non-str result means a tool ignored its -> str contract. Fail loud
        # (-> ERROR) rather than coerce and feed the model a corrupt observation.
        if not isinstance(result, str):
            raise TypeError(
                f"Tool {tool_name} returned {type(result).__name__}, expected str"
            )

        content = self._observation(result, charge=terminal_tool is None)

        return _ToolCallOutcome(
            call_log={
                "tool_name": tool_name,
                "args": tool_args,
                "timestamp": call_timestamp,
            },
            response_log={
                "tool_name": tool_name,
                "response_truncated": content[:MAX_LOGGED_RESPONSE_CHARS],
                "response_length": len(content),
                "timestamp": call_timestamp,
            },
            message={
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            },
            submitted=terminal_tool is not None,
            answer=answer,
            answer_tool=answer_tool,
        )

    def _create_completion(self, terminal_only: bool) -> Any:
        """Make one model API call, rejecting a response with no choices.

        Args:
            terminal_only: Offer only the terminal tools, for the final turn.

        Raises:
            RuntimeError: If the response carries no choices. Raised rather than
                returned so the retry wrapper treats it as a retriable failure,
                like a raised API exception.
        """
        tools = (
            self._config.terminal_tool_schemas
            if terminal_only
            else self._config.tool_schemas
        )
        response = self._config.model_api.chat.completions.create(
            model=self._config.model,
            messages=self._messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            **self._config.generation_params,
        )
        if not response.choices:
            raise RuntimeError(
                "Model API returned empty choices array. "
                "This may indicate rate limiting, content filtering, or API error."
            )
        return response

    def _call_model(self, terminal_only: bool) -> dict[str, Any]:
        """Call the model API with the current conversation, retrying failures.

        Args:
            terminal_only: Offer only the terminal tools, for the final turn.

        Returns:
            Model response as dict with 'role', 'content', and optional 'tool_calls'

        Raises:
            RuntimeError: If every attempt at the model API call fails.
        """
        try:
            response = RetryFunctionWrapper[Any](
                self._create_completion,
                max_attempts=self._config.api_max_attempts,
                retry_sleep_secs=self._config.api_retry_sleep_secs,
            )(terminal_only)
        except MaxAttemptsExceededError as e:
            raise RuntimeError(f"Model API call failed: {str(e)}") from e

        message = response.choices[0].message
        response_dict = {
            "role": "assistant",
            "content": message.content or "",
        }
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls: list[_ModelToolCall] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
            response_dict["tool_calls"] = tool_calls

        return response_dict
