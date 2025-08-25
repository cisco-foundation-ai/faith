# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from faith._internal.algo.graph import all_reachable_nodes, wcc_dict


def test_all_reachable_nodes() -> None:
    graph = {"A": ["B", "C"], "B": ["A", "D"], "C": ["A"], "D": ["E"], "E": ["D"]}
    assert sorted(all_reachable_nodes("A", graph)) == ["A", "B", "C", "D", "E"]
    assert sorted(all_reachable_nodes("B", graph)) == ["A", "B", "C", "D", "E"]
    assert sorted(all_reachable_nodes("C", graph)) == ["A", "B", "C", "D", "E"]
    assert sorted(all_reachable_nodes("D", graph)) == ["D", "E"]
    assert sorted(all_reachable_nodes("E", graph)) == ["D", "E"]


def test_wcc_dict() -> None:
    graph = {
        "A": ["B", "C"],
        "B": ["A", "D"],
        "C": ["A"],
        "D": ["E"],
        "E": ["D"],
        "F": [],
        "G": ["F"],
    }
    assert wcc_dict(graph) == {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 1, "G": 1}
