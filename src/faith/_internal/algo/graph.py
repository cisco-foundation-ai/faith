# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Functions for common graph algorithms and operations."""
from collections import defaultdict, deque


def all_reachable_nodes(start_node: str, graph: dict[str, list[str]]) -> list[str]:
    """Return all nodes reachable from `start_node` in the given `graph`."""
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)

    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)

    return list(visited)


def wcc_dict(graph: dict[str, list[str]]) -> dict[str, int]:
    """Map each node in `graph` to a weakly connected component (WCC) ID."""

    # Create a bidirectional graph from the directed graph representing all connections.
    bidirectional = defaultdict(list)
    for node, neighbors in graph.items():
        bidirectional[node].extend(neighbors)
        for neighbor in neighbors:
            bidirectional[neighbor].append(node)

    wcc_dict = {}
    wcc_id = 0
    for node in sorted(bidirectional):
        if node not in wcc_dict:
            for connected_node in all_reachable_nodes(node, bidirectional):
                wcc_dict[connected_node] = wcc_id
            wcc_id += 1
    return wcc_dict
