"""Streamlit component: side-by-side circuit comparison."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.visualizer import (
    build_layer_profile_figure,
    build_metrics_radar,
    build_side_by_side,
)


def render_metrics_row(comparison: dict[str, Any]) -> None:
    """Display top-level similarity metrics as a metric row."""
    m = comparison["metrics"]
    ov = m["feature_overlap"]
    st_m = m["structural_similarity"]
    ld = m["layer_distribution"]
    composite = m["composite_similarity"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite Similarity", f"{composite:.3f}")
    c2.metric("Feature Overlap (Jaccard)", f"{ov['jaccard']:.3f}")
    c3.metric("Structural Similarity", f"{st_m['composite_similarity']:.3f}")
    c4.metric("Layer EMD", f"{ld['emd']:.3f}", help="Earth-mover distance — lower = more similar")


def render_comparison_view(
    comparison: dict[str, Any],
    graph_a,
    graph_b,
) -> None:
    """Full comparison view: metrics, circuits, layer profiles, radar."""
    st.header(f"Comparing `{comparison['model_a']}` vs `{comparison['model_b']}`")
    prompt = comparison.get("prompt_text") or comparison.get("prompt_id", "")
    st.caption(f"Prompt: _{prompt}_")

    st.subheader("Similarity Metrics")
    render_metrics_row(comparison)

    # ---- radar chart ---------------------------------------------------
    col_radar, col_overlap = st.columns([1, 1])
    with col_radar:
        fig_radar = build_metrics_radar(comparison)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_overlap:
        ov = comparison["metrics"]["feature_overlap"]
        st.markdown("**Feature Overlap Detail**")
        st.dataframe(
            [
                {"Metric": "Shared features",   "Value": ov["shared"]},
                {"Metric": f"Only in {comparison['model_a']}", "Value": ov["only_model_1"]},
                {"Metric": f"Only in {comparison['model_b']}", "Value": ov["only_model_2"]},
                {"Metric": "Overlap coefficient", "Value": f"{ov['overlap_coefficient']:.3f}"},
            ],
            hide_index=True,
            use_container_width=True,
        )

    # ---- side-by-side circuits -----------------------------------------
    st.subheader("Circuit Graphs")
    shared_ids_a = {f["node_id"] for f in comparison.get("shared_features", [])}
    fig_both = build_side_by_side(graph_a, graph_b,
                                   shared_node_ids_a=shared_ids_a)
    st.plotly_chart(fig_both, use_container_width=True)

    # ---- layer profile -------------------------------------------------
    st.subheader("Layer Activation Profile")
    fig_layers = build_layer_profile_figure(
        comparison,
        model_a_label=comparison["model_a"],
        model_b_label=comparison["model_b"],
    )
    st.plotly_chart(fig_layers, use_container_width=True)

    # ---- path stats ----------------------------------------------------
    with st.expander("Path Statistics"):
        ps = comparison["metrics"]["path_statistics"]
        st.json({
            comparison["model_a"]: ps["model_1"],
            comparison["model_b"]: ps["model_2"],
            "depth_difference": ps["depth_difference"],
            "avg_path_length_difference": ps["avg_path_length_difference"],
        })

    # ---- top shared features -------------------------------------------
    with st.expander(f"Shared Features ({len(comparison['shared_features'])})"):
        st.dataframe(comparison["shared_features"], use_container_width=True)
