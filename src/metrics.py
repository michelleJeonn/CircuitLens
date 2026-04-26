"""Similarity metrics for comparing attribution graphs across models."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np

from .graph_processor import AttributionGraph
from .feature_extractor import feature_fingerprint, layer_activation_profile


# ---------------------------------------------------------------------------
# 1. Feature overlap
# ---------------------------------------------------------------------------

def compute_feature_overlap(g1: AttributionGraph, g2: AttributionGraph) -> dict[str, float]:
    """Overlap Coefficient and Jaccard index over (layer, feature_id) pairs.

    Overlap Coefficient = |A ∩ B| / min(|A|, |B|)  — robust when set sizes differ.
    Jaccard Index       = |A ∩ B| / |A ∪ B|
    """
    fp1 = feature_fingerprint(g1)
    fp2 = feature_fingerprint(g2)

    if not fp1 or not fp2:
        return {"overlap_coefficient": 0.0, "jaccard": 0.0,
                "shared": 0, "only_model_1": len(fp1), "only_model_2": len(fp2)}

    shared = fp1 & fp2
    union = fp1 | fp2

    return {
        "overlap_coefficient": len(shared) / min(len(fp1), len(fp2)),
        "jaccard": len(shared) / len(union),
        "shared": len(shared),
        "only_model_1": len(fp1 - fp2),
        "only_model_2": len(fp2 - fp1),
        "total_model_1": len(fp1),
        "total_model_2": len(fp2),
    }


# ---------------------------------------------------------------------------
# 2. Structural similarity
# ---------------------------------------------------------------------------

def compute_structural_similarity(g1: AttributionGraph, g2: AttributionGraph) -> dict[str, float]:
    """Graph-level structural statistics and a composite similarity score.

    Rather than computing exact graph edit distance (NP-hard for large graphs),
    we compare a set of summary statistics and derive a normalised similarity.

    Statistics compared:
      - density (edges / possible edges)
      - average clustering coefficient
      - proportion of nodes in the largest weakly-connected component
      - degree distribution similarity (cosine similarity of degree histograms)
    """
    stats1 = _graph_stats(g1.graph)
    stats2 = _graph_stats(g2.graph)

    scalar_keys = ["density", "avg_clustering", "lcc_fraction"]
    diffs = [abs(stats1[k] - stats2[k]) for k in scalar_keys]
    scalar_sim = 1.0 - np.mean(diffs)          # 1 = identical, 0 = maximally different

    degree_sim = _degree_histogram_cosine(g1.graph, g2.graph)

    composite = float(0.6 * scalar_sim + 0.4 * degree_sim)

    return {
        "composite_similarity": composite,
        "scalar_similarity": float(scalar_sim),
        "degree_distribution_similarity": float(degree_sim),
        "model_1": stats1,
        "model_2": stats2,
    }


def _graph_stats(G: nx.DiGraph) -> dict[str, float]:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = nx.density(G) if n > 1 else 0.0
    ug = G.to_undirected()
    avg_clustering = nx.average_clustering(ug) if n > 0 else 0.0
    components = list(nx.weakly_connected_components(G))
    lcc_fraction = max((len(c) for c in components), default=0) / n if n > 0 else 0.0
    return {
        "num_nodes": n,
        "num_edges": e,
        "density": density,
        "avg_clustering": avg_clustering,
        "lcc_fraction": lcc_fraction,
    }


def _degree_histogram_cosine(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
    """Cosine similarity between normalised degree histograms."""
    def _hist(G: nx.DiGraph) -> np.ndarray:
        degs = [d for _, d in G.degree()]
        if not degs:
            return np.zeros(1)
        max_d = max(degs)
        h = np.zeros(max_d + 1)
        for d in degs:
            h[d] += 1
        norm = np.linalg.norm(h)
        return h / norm if norm > 0 else h

    h1, h2 = _hist(G1), _hist(G2)
    # Pad to same length.
    length = max(len(h1), len(h2))
    h1 = np.pad(h1, (0, length - len(h1)))
    h2 = np.pad(h2, (0, length - len(h2)))
    dot = float(np.dot(h1, h2))
    return max(0.0, min(1.0, dot))


# ---------------------------------------------------------------------------
# 3. Path-based comparison
# ---------------------------------------------------------------------------

def compute_path_statistics(g1: AttributionGraph, g2: AttributionGraph) -> dict[str, Any]:
    """Compare average shortest-path length and circuit depth across models.

    We use a sampled approach (≤500 source/target pairs) to keep runtime
    manageable for large graphs.
    """
    stats1 = _path_stats(g1.graph)
    stats2 = _path_stats(g2.graph)

    depth_diff = abs(stats1["max_depth"] - stats2["max_depth"])
    avg_diff = abs(stats1["avg_path_length"] - stats2["avg_path_length"])

    return {
        "model_1": stats1,
        "model_2": stats2,
        "depth_difference": depth_diff,
        "avg_path_length_difference": avg_diff,
    }


def _path_stats(G: nx.DiGraph, sample_size: int = 500) -> dict[str, float]:
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {"avg_path_length": 0.0, "max_depth": 0, "num_paths_sampled": 0}

    rng = np.random.default_rng(42)
    pairs = list(zip(
        rng.choice(nodes, size=min(sample_size, n), replace=True),
        rng.choice(nodes, size=min(sample_size, n), replace=True),
    ))

    lengths = []
    for src, tgt in pairs:
        if src == tgt:
            continue
        try:
            lengths.append(nx.shortest_path_length(G, src, tgt))
        except nx.NetworkXNoPath:
            pass

    try:
        max_depth = max(nx.dag_longest_path_length(G), 0) if nx.is_directed_acyclic_graph(G) else 0
    except Exception:
        max_depth = 0

    return {
        "avg_path_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_depth": int(max_depth),
        "num_paths_sampled": len(lengths),
    }


# ---------------------------------------------------------------------------
# 4. Layer distribution
# ---------------------------------------------------------------------------

def compute_layer_distribution(g1: AttributionGraph, g2: AttributionGraph) -> dict[str, Any]:
    """Compare which layers are most active across models.

    Returns per-layer activation totals and an earth-mover's distance (EMD)
    between the two distributions.
    """
    prof1 = layer_activation_profile(g1)
    prof2 = layer_activation_profile(g2)

    all_layers = sorted(set(prof1) | set(prof2))

    dist1 = np.array([prof1.get(l, {}).get("total", 0.0) for l in all_layers])
    dist2 = np.array([prof2.get(l, {}).get("total", 0.0) for l in all_layers])

    # Normalise to probability distributions.
    def _normalise(v: np.ndarray) -> np.ndarray:
        s = v.sum()
        return v / s if s > 0 else v

    nd1, nd2 = _normalise(dist1), _normalise(dist2)

    # Earth-mover's distance over a 1-D ordered set of layers.
    emd = float(np.sum(np.abs(np.cumsum(nd1) - np.cumsum(nd2))))

    peak1 = all_layers[int(np.argmax(dist1))] if all_layers else None
    peak2 = all_layers[int(np.argmax(dist2))] if all_layers else None

    return {
        "emd": emd,                     # lower = more similar distributions
        "peak_layer_model_1": peak1,
        "peak_layer_model_2": peak2,
        "layers": all_layers,
        "distribution_model_1": nd1.tolist(),
        "distribution_model_2": nd2.tolist(),
        "profile_model_1": prof1,
        "profile_model_2": prof2,
    }


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_all_metrics(g1: AttributionGraph, g2: AttributionGraph) -> dict[str, Any]:
    """Compute all metrics and return them in a single dict."""
    overlap = compute_feature_overlap(g1, g2)
    structural = compute_structural_similarity(g1, g2)
    paths = compute_path_statistics(g1, g2)
    layers = compute_layer_distribution(g1, g2)

    # Simple composite: average of the three [0,1]-bounded scores.
    composite = float(np.mean([
        overlap["jaccard"],
        structural["composite_similarity"],
        max(0.0, 1.0 - layers["emd"]),   # EMD → similarity
    ]))

    return {
        "composite_similarity": composite,
        "feature_overlap": overlap,
        "structural_similarity": structural,
        "path_statistics": paths,
        "layer_distribution": layers,
        "model_1": g1.model_name,
        "model_2": g2.model_name,
        "prompt_id": g1.prompt_id,
    }
