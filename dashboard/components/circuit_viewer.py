"""Streamlit component: display a single attribution graph."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.graph_processor import AttributionGraph
from src.feature_extractor import get_top_features, group_by_layer
from src.visualizer import build_circuit_figure


def render_circuit_card(graph: AttributionGraph, title: str = "") -> None:
    """Render a circuit graph card with stats and interactive plot."""
    st.subheader(title or graph.model_name)

    summary = graph.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", summary["num_nodes"])
    c2.metric("Edges", summary["num_edges"])
    c3.metric("Layers", summary["num_layers"])
    c4.metric("Features", summary["num_features"])

    fig = build_circuit_figure(graph, title=graph.model_name, max_nodes=120)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Top 10 Features by Activation"):
        top = get_top_features(graph, top_k=10)
        rows = [
            {
                "Feature ID": n.feature_id,
                "Layer": n.layer,
                "Activation": round(n.activation, 5),
                "Label": n.label or n.node_id,
            }
            for n in top
        ]
        st.dataframe(rows, use_container_width=True)

    with st.expander("Layer Breakdown"):
        by_layer = group_by_layer(graph)
        layer_data = [
            {"Layer": layer, "# Features": len(nodes)}
            for layer, nodes in sorted(by_layer.items())
        ]
        st.dataframe(layer_data, use_container_width=True)
