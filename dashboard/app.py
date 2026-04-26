"""CircuitLens — Streamlit dashboard.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Allow imports from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuit_comparator import CircuitComparator
from src.graph_processor import GraphProcessor
from dashboard.components.circuit_viewer import render_circuit_card
from dashboard.components.comparison_view import render_comparison_view

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CircuitLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

GRAPHS_DIR = Path("data/graphs")
PROMPTS_FILE = Path("data/prompts.json")


@st.cache_data(show_spinner=False)
def load_prompts() -> list[dict]:
    if not PROMPTS_FILE.exists():
        return []
    with PROMPTS_FILE.open() as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def available_models() -> list[str]:
    if not GRAPHS_DIR.exists():
        return []
    return sorted(d.name for d in GRAPHS_DIR.iterdir() if d.is_dir())


@st.cache_data(show_spinner=False)
def available_prompts_for_model(model: str) -> list[str]:
    d = GRAPHS_DIR / model
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


@st.cache_data(show_spinner=False)
def available_prompts_for_models(model_a: str, model_b: str) -> list[str]:
    files_a = set(available_prompts_for_model(model_a))
    files_b = set(available_prompts_for_model(model_b))
    return sorted(files_a & files_b)


@st.cache_data(show_spinner="Loading graphs…")
def load_comparison(prompt_id: str, model_a: str, model_b: str) -> tuple:
    processor = GraphProcessor()
    ga = processor.load_graph(GRAPHS_DIR / model_a / f"{prompt_id}.json")
    gb = processor.load_graph(GRAPHS_DIR / model_b / f"{prompt_id}.json")
    comparator = CircuitComparator(ga, gb)
    return comparator.compare(), ga, gb


@st.cache_data(show_spinner="Loading graphs…")
def load_cross_prompt_comparison(model: str, prompt_a: str, prompt_b: str) -> tuple:
    processor = GraphProcessor()
    ga = processor.load_graph(GRAPHS_DIR / model / f"{prompt_a}.json")
    gb = processor.load_graph(GRAPHS_DIR / model / f"{prompt_b}.json")
    comparator = CircuitComparator(ga, gb)
    result = comparator.compare()
    # Relabel so the UI shows prompt IDs rather than "gemma vs gemma".
    result["model_a"] = f"{model} / {prompt_a}"
    result["model_b"] = f"{model} / {prompt_b}"
    return result, ga, gb


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🔬 CircuitLens")
st.sidebar.markdown("*Mechanistic interpretability — circuit comparison*")
st.sidebar.divider()

view_mode = st.sidebar.radio("View", ["Compare", "Single Circuit"])

models = available_models()

if not models:
    st.warning(
        "No attribution graphs found in `data/graphs/`. "
        "Run `notebooks/generate_graphs.ipynb` to generate them first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Compare view — cross-model when 2+ models exist, cross-prompt otherwise
# ---------------------------------------------------------------------------

if view_mode == "Compare":
    prompts = load_prompts()
    prompt_map = {p["id"]: p for p in prompts}

    def _prompt_label(pid: str) -> str:
        return f"{pid} — {prompt_map[pid]['prompt'][:55]}…" if pid in prompt_map else pid

    if len(models) >= 2:
        # ---- Cross-model mode ------------------------------------------------
        st.title("Cross-Model Circuit Comparison")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            model_a = st.selectbox("Model A", models, index=0, key="model_a")
        with col2:
            other_models = [m for m in models if m != model_a]
            model_b = st.selectbox("Model B", other_models, index=0, key="model_b")

        prompt_ids = available_prompts_for_models(model_a, model_b)
        if not prompt_ids:
            st.error(f"No shared prompts found for `{model_a}` and `{model_b}`.")
            st.stop()

        chosen_label = st.sidebar.selectbox("Prompt", [_prompt_label(p) for p in prompt_ids])
        chosen_id = prompt_ids[[_prompt_label(p) for p in prompt_ids].index(chosen_label)]

        with st.spinner("Computing comparison…"):
            comparison, graph_a, graph_b = load_comparison(chosen_id, model_a, model_b)

    else:
        # ---- Single-model cross-prompt mode ----------------------------------
        model = models[0]
        st.title(f"Cross-Prompt Circuit Comparison — {model}")
        st.info(
            f"Only **{model}** graphs are available. "
            "Comparing two different prompts on the same model. "
            "Add a second model's graphs to `data/graphs/` to enable cross-model comparison.",
            icon="ℹ️",
        )

        prompt_ids = available_prompts_for_model(model)
        if len(prompt_ids) < 2:
            st.error(f"Need at least 2 prompt graphs in `data/graphs/{model}/`. Generate more prompts first.")
            st.stop()

        labels = [_prompt_label(p) for p in prompt_ids]
        col1, col2 = st.sidebar.columns(2)
        with col1:
            label_a = st.selectbox("Prompt A", labels, index=0, key="prompt_a")
        with col2:
            other_labels = [l for l in labels if l != label_a]
            label_b = st.selectbox("Prompt B", other_labels, index=min(1, len(other_labels) - 1), key="prompt_b")

        pid_a = prompt_ids[labels.index(label_a)]
        pid_b = prompt_ids[[_prompt_label(p) for p in prompt_ids].index(label_b)]

        with st.spinner("Computing comparison…"):
            comparison, graph_a, graph_b = load_cross_prompt_comparison(model, pid_a, pid_b)

    render_comparison_view(comparison, graph_a, graph_b)

# ---------------------------------------------------------------------------
# Single Circuit view
# ---------------------------------------------------------------------------

else:
    st.title("Single Circuit Inspector")

    model = st.sidebar.selectbox("Model", models, key="single_model")
    model_dir = GRAPHS_DIR / model
    prompt_ids = sorted(p.stem for p in model_dir.glob("*.json")) if model_dir.exists() else []

    if not prompt_ids:
        st.error(f"No graphs found for `{model}`.")
        st.stop()

    prompts = load_prompts()
    prompt_map = {p["id"]: p for p in prompts}
    prompt_labels = [
        f"{pid} — {prompt_map[pid]['prompt'][:60]}…" if pid in prompt_map else pid
        for pid in prompt_ids
    ]
    chosen_label = st.sidebar.selectbox("Prompt", prompt_labels)
    chosen_id = prompt_ids[prompt_labels.index(chosen_label)]

    processor = GraphProcessor()
    graph = processor.load_graph(model_dir / f"{chosen_id}.json")
    render_circuit_card(graph, title=f"{model} — {chosen_id}")
