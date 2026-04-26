"""Build Plotly figures from attribution graphs and comparison results."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .graph_processor import AttributionGraph


# ---------------------------------------------------------------------------
# Spring-layout circuit graph
# ---------------------------------------------------------------------------

def build_circuit_figure(
    graph: AttributionGraph,
    title: str = "",
    max_nodes: int = 150,
    highlight_nodes: set[str] | None = None,
) -> go.Figure:
    """Render an attribution graph as an interactive Plotly network."""
    G = graph.graph

    # Subsample very large graphs to keep the browser responsive.
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(
            G.nodes(), key=lambda n: abs(G.nodes[n]["info"].activation), reverse=True
        )[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    pos = nx.spring_layout(G, seed=42, k=2 / max(G.number_of_nodes() ** 0.5, 1))

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="#888"),
        hoverinfo="none",
    )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for nid in G.nodes():
        x, y = pos[nid]
        info = G.nodes[nid]["info"]
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            f"ID: {nid}<br>Type: {info.node_type}<br>Layer: {info.layer}"
            f"<br>Feature: {info.feature_id}<br>Activation: {info.activation:.4f}"
        )
        act = abs(info.activation)
        node_color.append(act)
        size = 6 + 14 * (act / (max(node_color or [1.0]) or 1.0))
        node_size.append(size)
        if highlight_nodes and nid in highlight_nodes:
            node_size[-1] = size * 1.8

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_color,
            size=node_size,
            colorbar=dict(thickness=12, title="Activation", xanchor="left"),
            line=dict(width=0.5, color="#fff"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title or graph.model_name,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=10, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Layer activation profile bar chart
# ---------------------------------------------------------------------------

def build_layer_profile_figure(
    comparison: dict[str, Any],
    model_a_label: str = "Model A",
    model_b_label: str = "Model B",
) -> go.Figure:
    """Side-by-side bar chart of per-layer activation totals."""
    ld = comparison["metrics"]["layer_distribution"]
    layers = ld["layers"]
    dist_a = ld["distribution_model_1"]
    dist_b = ld["distribution_model_2"]

    fig = go.Figure(data=[
        go.Bar(name=model_a_label, x=layers, y=dist_a, marker_color="#636EFA"),
        go.Bar(name=model_b_label, x=layers, y=dist_b, marker_color="#EF553B"),
    ])
    fig.update_layout(
        barmode="group",
        title="Layer Activation Distribution",
        xaxis_title="Layer",
        yaxis_title="Normalised Activation",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


# ---------------------------------------------------------------------------
# Metric summary radar chart
# ---------------------------------------------------------------------------

def build_metrics_radar(comparison: dict[str, Any]) -> go.Figure:
    """Radar chart showing the four similarity dimensions."""
    metrics = comparison["metrics"]
    ov = metrics["feature_overlap"]
    st = metrics["structural_similarity"]
    ld = metrics["layer_distribution"]

    categories = ["Feature Overlap", "Structural", "Layer Similarity", "Composite"]
    values = [
        ov["jaccard"],
        st["composite_similarity"],
        max(0.0, 1.0 - ld["emd"]),
        metrics["composite_similarity"],
    ]
    values += [values[0]]          # close the polygon
    categories += [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        line_color="#7B2FBE",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Similarity Profile",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Side-by-side comparison figure
# ---------------------------------------------------------------------------

def build_side_by_side(
    graph_a: AttributionGraph,
    graph_b: AttributionGraph,
    shared_node_ids_a: set[str] | None = None,
    shared_node_ids_b: set[str] | None = None,
) -> go.Figure:
    """Two circuit graphs displayed side by side in one Plotly figure."""
    fig_a = build_circuit_figure(graph_a, title=graph_a.model_name, highlight_nodes=shared_node_ids_a)
    fig_b = build_circuit_figure(graph_b, title=graph_b.model_name, highlight_nodes=shared_node_ids_b)

    combined = make_subplots(rows=1, cols=2, subplot_titles=[graph_a.model_name, graph_b.model_name])
    for trace in fig_a.data:
        combined.add_trace(trace, row=1, col=1)
    for trace in fig_b.data:
        combined.add_trace(trace, row=1, col=2)

    combined.update_layout(
        showlegend=False,
        hovermode="closest",
        title="Circuit Comparison",
        height=520,
    )
    combined.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    combined.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return combined
