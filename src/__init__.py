"""CircuitLens — mechanistic interpretability circuit comparison across LLMs."""

from .graph_processor import GraphProcessor, AttributionGraph
from .feature_extractor import (
    get_active_features,
    get_top_features,
    group_by_layer,
    layer_activation_profile,
    feature_fingerprint,
    extract_summary,
)
from .circuit_comparator import CircuitComparator
from .metrics import (
    compute_feature_overlap,
    compute_structural_similarity,
    compute_path_statistics,
    compute_layer_distribution,
)

__all__ = [
    "GraphProcessor",
    "AttributionGraph",
    "get_active_features",
    "get_top_features",
    "group_by_layer",
    "layer_activation_profile",
    "feature_fingerprint",
    "extract_summary",
    "CircuitComparator",
    "compute_feature_overlap",
    "compute_structural_similarity",
    "compute_path_statistics",
    "compute_layer_distribution",
]
