"""Unit tests for src/metrics.py using synthetic attribution graphs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from src.graph_processor import AttributionGraph, GraphProcessor, NodeInfo
from src.metrics import (
    compute_all_metrics,
    compute_feature_overlap,
    compute_layer_distribution,
    compute_path_statistics,
    compute_structural_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(
    num_features: int = 10,
    num_layers: int = 3,
    num_edges: int = 15,
    model_name: str = "test_model",
    prompt_id: str = "test_prompt",
    seed: int = 0,
) -> AttributionGraph:
    """Create a minimal synthetic AttributionGraph for testing."""
    import random
    rng = random.Random(seed)

    G = nx.DiGraph()
    node_ids = []
    for i in range(num_features):
        nid = f"feat_{i}"
        layer = i % num_layers
        info = NodeInfo(
            node_id=nid,
            node_type="feature",
            layer=layer,
            position=i,
            feature_id=i,
            activation=rng.uniform(-1, 1),
            label=f"feature_{i}",
        )
        G.add_node(nid, info=info)
        node_ids.append(nid)

    for _ in range(num_edges):
        u, v = rng.sample(node_ids, 2)
        from src.graph_processor import EdgeInfo
        G.add_edge(u, v, info=EdgeInfo(u, v, rng.uniform(0.01, 1.0)), weight=rng.uniform(0.01, 1.0))

    return AttributionGraph(G, model_name=model_name, prompt_id=prompt_id, prompt_text="test")


def _make_graph_from_json(data: dict, model_name: str = "model", prompt_id: str = "p1") -> AttributionGraph:
    """Write data to a temp JSON file and load via GraphProcessor."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp_path = Path(f.name)
    processor = GraphProcessor()
    return processor.load_graph(tmp_path, model_name=model_name, prompt_id=prompt_id)


# ---------------------------------------------------------------------------
# GraphProcessor tests
# ---------------------------------------------------------------------------

class TestGraphProcessor:
    def test_load_minimal_json(self):
        data = {
            "nodes": [
                {"node_id": "n1", "type": "feature", "layer": 0, "feature_id": 1, "activation": 0.5},
                {"node_id": "n2", "type": "feature", "layer": 1, "feature_id": 2, "activation": -0.3},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "weight": 0.8},
            ],
        }
        g = _make_graph_from_json(data)
        assert g.graph.number_of_nodes() == 2
        assert g.graph.number_of_edges() == 1

    def test_dangling_edge_creates_node(self):
        data = {
            "nodes": [{"node_id": "n1", "type": "feature", "layer": 0, "activation": 1.0}],
            "edges": [{"source": "n1", "target": "ghost", "weight": 0.5}],
        }
        g = _make_graph_from_json(data)
        assert "ghost" in g.graph.nodes

    def test_alternate_key_names(self):
        data = {
            "nodes": [{"id": "a", "kind": "token", "pos": 3, "act": 0.9}],
            "edges": [{"src": "a", "dst": "a", "effect": 0.1}],
        }
        g = _make_graph_from_json(data)
        assert g.graph.number_of_nodes() == 1

    def test_empty_graph(self):
        data = {"nodes": [], "edges": []}
        g = _make_graph_from_json(data)
        assert g.graph.number_of_nodes() == 0

    def test_metadata_extracted(self):
        data = {
            "nodes": [],
            "edges": [],
            "metadata": {"prompt": "hello world", "model": "test"},
        }
        g = _make_graph_from_json(data, model_name="", prompt_id="p1")
        assert g.prompt_text == "hello world"


# ---------------------------------------------------------------------------
# Feature overlap tests
# ---------------------------------------------------------------------------

class TestFeatureOverlap:
    def test_identical_graphs(self):
        g = _make_graph(seed=0)
        result = compute_feature_overlap(g, g)
        assert result["jaccard"] == pytest.approx(1.0)
        assert result["overlap_coefficient"] == pytest.approx(1.0)
        assert result["shared"] == result["total_model_1"]

    def test_disjoint_graphs(self):
        g1 = _make_graph(num_features=5, seed=1)
        g2 = _make_graph(num_features=5, seed=2)
        # Give each graph unique feature_ids so they can't overlap.
        for i, (nid, data) in enumerate(g1.graph.nodes(data=True)):
            data["info"].feature_id = i
        for i, (nid, data) in enumerate(g2.graph.nodes(data=True)):
            data["info"].feature_id = i + 100
        result = compute_feature_overlap(g1, g2)
        assert result["shared"] == 0
        assert result["jaccard"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        g1 = _make_graph(num_features=10, seed=3)
        g2 = _make_graph(num_features=10, seed=4)
        # Force half the features to match.
        for nid, data in list(g2.graph.nodes(data=True))[:5]:
            data["info"].feature_id = list(g1.graph.nodes(data=True))[0][1]["info"].feature_id
        result = compute_feature_overlap(g1, g2)
        assert 0.0 <= result["jaccard"] <= 1.0
        assert 0.0 <= result["overlap_coefficient"] <= 1.0

    def test_empty_graphs(self):
        g1 = _make_graph(num_features=0, num_edges=0)
        g2 = _make_graph(num_features=0, num_edges=0)
        result = compute_feature_overlap(g1, g2)
        assert result["jaccard"] == 0.0


# ---------------------------------------------------------------------------
# Structural similarity tests
# ---------------------------------------------------------------------------

class TestStructuralSimilarity:
    def test_identical(self):
        g = _make_graph(seed=5)
        result = compute_structural_similarity(g, g)
        assert result["composite_similarity"] >= 0.9

    def test_returns_bounded_score(self):
        g1 = _make_graph(seed=6)
        g2 = _make_graph(num_features=20, num_edges=30, seed=7)
        result = compute_structural_similarity(g1, g2)
        assert 0.0 <= result["composite_similarity"] <= 1.0

    def test_model_stats_present(self):
        g1 = _make_graph(seed=8)
        g2 = _make_graph(seed=9)
        result = compute_structural_similarity(g1, g2)
        assert "num_nodes" in result["model_1"]
        assert "density" in result["model_2"]


# ---------------------------------------------------------------------------
# Path statistics tests
# ---------------------------------------------------------------------------

class TestPathStatistics:
    def test_returns_dict(self):
        g1 = _make_graph(seed=10)
        g2 = _make_graph(seed=11)
        result = compute_path_statistics(g1, g2)
        assert "model_1" in result and "model_2" in result

    def test_nonnegative_values(self):
        g1 = _make_graph(seed=12)
        g2 = _make_graph(seed=13)
        result = compute_path_statistics(g1, g2)
        assert result["depth_difference"] >= 0
        assert result["avg_path_length_difference"] >= 0

    def test_empty_graph(self):
        g1 = _make_graph(num_features=0, num_edges=0)
        g2 = _make_graph(num_features=0, num_edges=0)
        result = compute_path_statistics(g1, g2)
        assert result["model_1"]["avg_path_length"] == 0.0


# ---------------------------------------------------------------------------
# Layer distribution tests
# ---------------------------------------------------------------------------

class TestLayerDistribution:
    def test_emd_identical(self):
        g = _make_graph(seed=14)
        result = compute_layer_distribution(g, g)
        assert result["emd"] == pytest.approx(0.0, abs=1e-6)

    def test_emd_range(self):
        g1 = _make_graph(seed=15)
        g2 = _make_graph(seed=16)
        result = compute_layer_distribution(g1, g2)
        assert result["emd"] >= 0.0

    def test_distributions_sum_to_one(self):
        g1 = _make_graph(seed=17)
        g2 = _make_graph(seed=18)
        result = compute_layer_distribution(g1, g2)
        import math
        if result["distribution_model_1"]:
            assert math.isclose(sum(result["distribution_model_1"]), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# compute_all_metrics integration test
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_all_keys_present(self):
        g1 = _make_graph(seed=20)
        g2 = _make_graph(seed=21)
        result = compute_all_metrics(g1, g2)
        for key in ("composite_similarity", "feature_overlap", "structural_similarity",
                    "path_statistics", "layer_distribution"):
            assert key in result

    def test_composite_bounded(self):
        g1 = _make_graph(seed=22)
        g2 = _make_graph(seed=23)
        result = compute_all_metrics(g1, g2)
        assert 0.0 <= result["composite_similarity"] <= 1.0
