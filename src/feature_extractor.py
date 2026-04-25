"""Extract and aggregate features from attribution graphs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .graph_processor import AttributionGraph, NodeInfo


def get_active_features(
    graph: AttributionGraph,
    activation_threshold: float = 0.0,
) -> list[NodeInfo]:
    """Return feature nodes with |activation| above threshold."""
    return [
        n for n in graph.feature_nodes
        if abs(n.activation) > activation_threshold
    ]


def group_by_layer(graph: AttributionGraph) -> dict[int, list[NodeInfo]]:
    """Map layer index → list of feature nodes in that layer."""
    groups: dict[int, list[NodeInfo]] = defaultdict(list)
    for node in graph.feature_nodes:
        if node.layer is not None:
            groups[node.layer].append(node)
    return dict(sorted(groups.items()))


def get_top_features(
    graph: AttributionGraph,
    top_k: int = 20,
    by: str = "activation",
) -> list[NodeInfo]:
    """Return the top-k features ranked by activation magnitude or in-degree."""
    if by == "activation":
        return sorted(graph.feature_nodes, key=lambda n: abs(n.activation), reverse=True)[:top_k]
    if by == "indegree":
        deg = dict(graph.graph.in_degree())
        return sorted(graph.feature_nodes, key=lambda n: deg.get(n.node_id, 0), reverse=True)[:top_k]
    if by == "outdegree":
        deg = dict(graph.graph.out_degree())
        return sorted(graph.feature_nodes, key=lambda n: deg.get(n.node_id, 0), reverse=True)[:top_k]
    raise ValueError(f"Unknown ranking criterion: {by!r}. Choose 'activation', 'indegree', or 'outdegree'.")


def layer_activation_profile(graph: AttributionGraph) -> dict[int, dict[str, float]]:
    """Summarize mean/max/count of activations per layer."""
    by_layer = group_by_layer(graph)
    profile: dict[int, dict[str, float]] = {}
    for layer, nodes in by_layer.items():
        acts = np.array([abs(n.activation) for n in nodes])
        profile[layer] = {
            "mean": float(acts.mean()),
            "max": float(acts.max()),
            "count": float(len(nodes)),
            "total": float(acts.sum()),
        }
    return profile


def feature_fingerprint(graph: AttributionGraph) -> set[tuple[int | None, int | None]]:
    """Canonical (layer, feature_id) pairs — used for cross-model matching."""
    return {
        (n.layer, n.feature_id)
        for n in graph.feature_nodes
        if n.feature_id is not None
    }


def extract_summary(graph: AttributionGraph) -> dict[str, Any]:
    """Return a flat dict of descriptive statistics for a graph."""
    acts = np.array([abs(n.activation) for n in graph.feature_nodes]) if graph.feature_nodes else np.array([0.0])
    return {
        **graph.summary(),
        "mean_activation": float(acts.mean()),
        "max_activation": float(acts.max()),
        "active_layers": sorted({n.layer for n in graph.feature_nodes if n.layer is not None}),
        "unique_features": len({n.feature_id for n in graph.feature_nodes if n.feature_id is not None}),
    }
