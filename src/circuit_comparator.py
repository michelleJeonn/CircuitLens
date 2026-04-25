"""High-level interface for comparing attribution graphs across models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .graph_processor import AttributionGraph, GraphProcessor
from .feature_extractor import (
    extract_summary,
    get_active_features,
    get_top_features,
    group_by_layer,
)
from .metrics import compute_all_metrics


class CircuitComparator:
    """Compare attribution graphs for the same prompt across two models.

    Usage::

        comparator = CircuitComparator(gemma_graph, llama_graph)
        results = comparator.compare()
    """

    def __init__(self, graph_a: AttributionGraph, graph_b: AttributionGraph) -> None:
        self.graph_a = graph_a
        self.graph_b = graph_b

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def compare(
        self,
        activation_threshold: float = 0.0,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """Run the full comparison and return a structured result dict."""
        metrics = compute_all_metrics(self.graph_a, self.graph_b)

        shared_features, unique_a, unique_b = self._align_features()

        top_a = get_top_features(self.graph_a, top_k)
        top_b = get_top_features(self.graph_b, top_k)

        active_a = get_active_features(self.graph_a, activation_threshold)
        active_b = get_active_features(self.graph_b, activation_threshold)

        return {
            "prompt_id": self.graph_a.prompt_id,
            "prompt_text": self.graph_a.prompt_text or self.graph_b.prompt_text,
            "model_a": self.graph_a.model_name,
            "model_b": self.graph_b.model_name,
            "metrics": metrics,
            "summary_a": extract_summary(self.graph_a),
            "summary_b": extract_summary(self.graph_b),
            "shared_features": shared_features,
            "unique_to_a": unique_a,
            "unique_to_b": unique_b,
            "top_features_a": self._serialise_nodes(top_a),
            "top_features_b": self._serialise_nodes(top_b),
            "active_count_a": len(active_a),
            "active_count_b": len(active_b),
            "layer_breakdown_a": self._serialise_layer_breakdown(group_by_layer(self.graph_a)),
            "layer_breakdown_b": self._serialise_layer_breakdown(group_by_layer(self.graph_b)),
        }

    def feature_overlap_report(self) -> str:
        """Human-readable overlap report."""
        from .metrics import compute_feature_overlap
        ov = compute_feature_overlap(self.graph_a, self.graph_b)
        lines = [
            f"Feature Overlap: {self.graph_a.model_name} vs {self.graph_b.model_name}",
            f"  Shared features   : {ov['shared']}",
            f"  Only in {self.graph_a.model_name:<12}: {ov['only_model_1']}",
            f"  Only in {self.graph_b.model_name:<12}: {ov['only_model_2']}",
            f"  Overlap coefficient: {ov['overlap_coefficient']:.3f}",
            f"  Jaccard index      : {ov['jaccard']:.3f}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Class method for loading from disk                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_files(
        cls,
        prompt_id: str,
        model_a: str,
        model_b: str,
        graphs_dir: str | Path = "data/graphs",
    ) -> "CircuitComparator":
        """Construct directly from JSON files on disk."""
        processor = GraphProcessor()
        graphs_dir = Path(graphs_dir)
        ga = processor.load_graph(graphs_dir / model_a / f"{prompt_id}.json")
        gb = processor.load_graph(graphs_dir / model_b / f"{prompt_id}.json")
        return cls(ga, gb)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _align_features(
        self,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Split features into shared vs unique sets using (layer, feature_id) keys."""
        from .feature_extractor import feature_fingerprint

        fp_a = feature_fingerprint(self.graph_a)
        fp_b = feature_fingerprint(self.graph_b)

        shared_keys = fp_a & fp_b
        unique_a_keys = fp_a - fp_b
        unique_b_keys = fp_b - fp_a

        def _nodes_for_keys(graph: AttributionGraph, keys: set) -> list[dict]:
            out = []
            for node in graph.feature_nodes:
                if (node.layer, node.feature_id) in keys:
                    out.append({
                        "node_id": node.node_id,
                        "layer": node.layer,
                        "feature_id": node.feature_id,
                        "activation": node.activation,
                        "label": node.label,
                    })
            return out

        return (
            _nodes_for_keys(self.graph_a, shared_keys),
            _nodes_for_keys(self.graph_a, unique_a_keys),
            _nodes_for_keys(self.graph_b, unique_b_keys),
        )

    @staticmethod
    def _serialise_nodes(nodes) -> list[dict]:
        return [
            {
                "node_id": n.node_id,
                "layer": n.layer,
                "feature_id": n.feature_id,
                "activation": n.activation,
                "label": n.label,
            }
            for n in nodes
        ]

    @staticmethod
    def _serialise_layer_breakdown(by_layer: dict) -> dict[int, dict]:
        import numpy as np
        out = {}
        for layer, nodes in by_layer.items():
            acts = [abs(n.activation) for n in nodes]
            out[layer] = {
                "count": len(nodes),
                "mean_activation": float(np.mean(acts)) if acts else 0.0,
                "max_activation": float(np.max(acts)) if acts else 0.0,
            }
        return out
