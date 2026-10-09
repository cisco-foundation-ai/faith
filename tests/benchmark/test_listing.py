# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from numbers import Number
from pathlib import Path
from typing import cast
from unittest.mock import patch

from pytest_unordered import unordered

from faith._internal.io.resources import benchmarks_root
from faith._internal.metrics.aggregations import agg_trial_stats
from faith._types.record.score import Score
from faith.benchmark.config import load_config_from_path
from faith.benchmark.listing import (
    benchmark_choices,
    benchmark_groups,
    choices_to_benchmarks,
    find_benchmarks,
)
from faith.benchmark.scores.domain_specific import DomainSpecificScore


def test_benchmark_choices() -> None:
    assert benchmark_choices() == [
        "all",
        "all-reasoning",
        "all-security",
        "cti-reasoning",
        "!cti-reasoning",
        "ctibench-ate",
        "!ctibench-ate",
        "ctibench-mcqa",
        "!ctibench-mcqa",
        "ctibench-rcm",
        "!ctibench-rcm",
        "ctibench-taa",
        "!ctibench-taa",
        "ctibench-vsp",
        "!ctibench-vsp",
        "cwe-prediction",
        "!cwe-prediction",
        "cybermetric-10000",
        "!cybermetric-10000",
        "cybermetric-2000",
        "!cybermetric-2000",
        "cybermetric-500",
        "!cybermetric-500",
        "cybermetric-80",
        "!cybermetric-80",
        "mmlu-all",
        "!mmlu-all",
        "mmlu-security",
        "!mmlu-security",
        "secbench-mcqa-eng",
        "!secbench-mcqa-eng",
        "secbench-mcqa-eng-reasoning",
        "!secbench-mcqa-eng-reasoning",
        "seceval",
        "!seceval",
        "vulnerability-localization-a",
        "!vulnerability-localization-a",
        "vulnerability-localization-b",
        "!vulnerability-localization-b",
    ]


def benchmark_subpath(sub_paths: list[str]) -> list[Path]:
    """Helper to convert sub-path strings to full benchmark paths."""
    return [benchmarks_root() / p for p in sub_paths]


def test_choices_to_benchmarks() -> None:
    assert choices_to_benchmarks([]) == []
    assert choices_to_benchmarks(["all"]) == benchmark_subpath(
        [
            "cti-reasoning",
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cwe-prediction",
            "cybermetric-10000",
            "cybermetric-2000",
            "cybermetric-500",
            "cybermetric-80",
            "mmlu-all",
            "mmlu-security",
            "secbench-mcqa-eng",
            "secbench-mcqa-eng-reasoning",
            "seceval",
        ]
    )
    assert choices_to_benchmarks(["all-reasoning"]) == benchmark_subpath(
        [
            "cti-reasoning",
            "secbench-mcqa-eng-reasoning",
        ]
    )
    assert choices_to_benchmarks(["all-security"]) == benchmark_subpath(
        [
            "cti-reasoning",
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cwe-prediction",
            "cybermetric-10000",
            "cybermetric-2000",
            "cybermetric-500",
            "cybermetric-80",
            "mmlu-security",
            "secbench-mcqa-eng",
            "secbench-mcqa-eng-reasoning",
            "seceval",
        ]
    )
    assert choices_to_benchmarks(["mmlu-all", "all"]) == benchmark_subpath(
        [
            "mmlu-all",
            "cti-reasoning",
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cwe-prediction",
            "cybermetric-10000",
            "cybermetric-2000",
            "cybermetric-500",
            "cybermetric-80",
            "mmlu-security",
            "secbench-mcqa-eng",
            "secbench-mcqa-eng-reasoning",
            "seceval",
        ]
    )
    assert choices_to_benchmarks(["!mmlu-all", "all", "!seceval"]) == benchmark_subpath(
        [
            "cti-reasoning",
            "ctibench-ate",
            "ctibench-mcqa",
            "ctibench-rcm",
            "ctibench-vsp",
            "cwe-prediction",
            "cybermetric-10000",
            "cybermetric-2000",
            "cybermetric-500",
            "cybermetric-80",
            "mmlu-security",
            "secbench-mcqa-eng",
            "secbench-mcqa-eng-reasoning",
        ]
    )
    assert choices_to_benchmarks(["mmlu-security"]) == benchmark_subpath(
        ["mmlu-security"]
    )


def test_find_benchmarks() -> None:
    # Test that all of the built-in benchmarks are found.
    assert find_benchmarks(benchmarks_root()) == unordered(
        benchmark_subpath(
            [
                "cti-reasoning",
                "ctibench-ate",
                "ctibench-mcqa",
                "ctibench-rcm",
                "ctibench-vsp",
                "cwe-prediction",
                "cybermetric-10000",
                "cybermetric-2000",
                "cybermetric-500",
                "cybermetric-80",
                "mmlu-all",
                "mmlu-security",
                "secbench-mcqa-eng",
                "secbench-mcqa-eng-reasoning",
                "seceval",
                "vulnerability-localization-a",
                "vulnerability-localization-b",
            ]
        )
    )

    # Test that the correct external benchmarks are found. find_benchmarks is
    # unfiltered by type, so an enabled agentic benchmark is still listed here
    # (i.e. individually selectable) even though it is excluded from the groups.
    assert find_benchmarks(Path(__file__).parent / "testdata") == unordered(
        [
            Path(__file__).parent / "testdata" / "bench-a",
            Path(__file__).parent / "testdata" / "bench-agentic",
            Path(__file__).parent / "testdata" / "sub" / "bench-c",
        ]
    )


def test_every_group_excludes_agentic_benchmarks() -> None:
    """No group sweeps in an agentic benchmark, but it stays selectable by name.

    Covers every group rather than a sampled one, so a group added later without
    the exclusion fails here.
    """
    testdata = Path(__file__).parent / "testdata"
    # bench-agentic is categorized security + reasoning, so it would land in
    # every group if it were not excluded.
    with patch("faith.benchmark.listing.benchmarks_root", return_value=testdata):
        for group in benchmark_groups():
            selected = choices_to_benchmarks([group])
            assert testdata / "bench-agentic" not in selected, group

        assert choices_to_benchmarks(["all"]) == [
            testdata / "bench-a",
            testdata / "sub" / "bench-c",
        ]
        assert choices_to_benchmarks(["bench-agentic"]) == [testdata / "bench-agentic"]


def test_all_benchmarks_have_a_primary_metric_that_resolves() -> None:
    """Ensure every benchmark's primary_metric names a statistic that gets emitted.

    A name that resolves to nothing is silent: `is_primary` is simply set on no row,
    so the benchmark's BigQuery export has no primary metric and nothing complains.

    A primary_metric is `<metric>.<trial_stat>`, where the metric a score fn
    contributes is `<aggregate_key>_<score_fn_name>` — so a score fn's own name is
    never the metric, and `file_f1.mean` names nothing while `mean_file_f1.mean`
    names the mean-across-tasks aggregated across trials.

    QA's structural metrics (`accuracy`, `subject_weighted_accuracy`) are computed in
    code rather than declared, so only the trial statistic is checkable for those.
    """
    # Read off the real aggregator rather than restated, so a stat added there is
    # accepted here without editing this test.
    sampled = agg_trial_stats(cast(list[Number], [0.0, 1.0]))
    assert isinstance(sampled, dict)
    trial_stats = set(sampled)

    for benchmark_path in find_benchmarks(benchmarks_root()):
        output_processing = load_config_from_path(benchmark_path).output_processing
        primary_metric = output_processing.primary_metric
        assert primary_metric, f"{benchmark_path.name} must set a 'primary_metric'."

        metric, _, trial_stat = primary_metric.rpartition(".")
        assert trial_stat in trial_stats, (
            f"{benchmark_path.name} names trial statistic '{trial_stat}', which is "
            f"not one of {sorted(trial_stats)}."
        )

        score_fns = DomainSpecificScore.from_configs(**output_processing.score_fns)
        emitted = {
            f"{aggregate_key}_{name}"
            for name, score_fn in score_fns.items()
            for aggregate_key in score_fn.aggregate([Score(raw_value=1.0, value=1.0)])
        }
        if any(metric == name or metric.endswith(f"_{name}") for name in score_fns):
            assert metric in emitted, (
                f"{benchmark_path.name} names metric '{metric}', which no score fn "
                f"emits; its score fns emit {sorted(emitted)}."
            )


def test_every_agentic_benchmark_briefs_its_agent_on_its_own_tools() -> None:
    """An agentic system prompt must name every tool the benchmark enables.

    Guards the copy-paste that drops half a prompt, and the rename that leaves prose
    naming a tool the model was never offered — the prompt says `terminal`
    because its tool is named `terminal`. The tool list is read from the same config,
    so this holds for a benchmark whose toolset changes without editing this test.
    """
    for benchmark_path in find_benchmarks(benchmarks_root()):
        agentic = load_config_from_path(benchmark_path).agentic_config
        if agentic is None:
            continue
        prompt = agentic.system_prompt
        assert prompt, f"Agentic benchmark {benchmark_path.name} sets no system_prompt."
        assert [t for t in agentic.tools if t not in prompt] == [], (
            f"{benchmark_path.name}'s system_prompt does not name every tool it "
            f"enables: {agentic.tools}"
        )


def test_each_vulnloc_phase_declares_the_scores_the_reference_aggregates() -> None:
    """Phase A needs both: `file_f1` alone scores a give-up and a wrong guess 0.0."""
    for name, score_fns in (
        ("vulnerability-localization-a", {"file_f1", "abstain_rate"}),
        ("vulnerability-localization-b", {"true_negative_rate"}),
    ):
        output_processing = load_config_from_path(
            benchmarks_root() / name
        ).output_processing
        assert set(output_processing.score_fns) == score_fns, name


def test_both_vulnloc_phases_require_read_only_workspaces() -> None:
    for name in ("vulnerability-localization-a", "vulnerability-localization-b"):
        agentic = load_config_from_path(benchmarks_root() / name).agentic_config
        assert agentic is not None and agentic.read_only_workspace is True


def test_the_prompt_states_the_tool_budget_it_is_configured_with() -> None:
    """The prompt spends its call budget in prose, so the two must agree.

    The budget is stated twice ("up to 15 terminal calls", "at least 7-10"), so
    raising max_tool_calls without rewording leaves the agent self-limiting to a
    stale number. Derived from the limit rather than transcribed, so the yaml stays
    the single place the budget is written.
    """
    for name in ("vulnerability-localization-a", "vulnerability-localization-b"):
        agentic = load_config_from_path(benchmarks_root() / name).agentic_config
        assert agentic is not None and agentic.system_prompt
        budget = agentic.limits.max_tool_calls
        assert (
            f"up to {budget} terminal calls" in agentic.system_prompt
        ), f"{name}'s prompt does not state its own budget of {budget} tool calls."
