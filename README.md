# CircuitLens

**Mechanistic interpretability circuit comparison** — generate, compare, and visualize attribution graphs for Gemma-2-2B and Llama-3.2-1B on identical prompts.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## What it does

CircuitLens answers the question: *when two language models produce the same output, do they use similar internal circuits?*

For a set of test prompts (factual recall, arithmetic, logical reasoning, etc.) it:

1. Runs [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) on both models to produce attribution graphs
2. Extracts features, layer assignments, and edge weights from each graph
3. Computes four similarity metrics comparing the graphs
4. Displays everything in an interactive Streamlit dashboard

---

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/michelleJeonn/CircuitLens.git
cd CircuitLens

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Log in to Hugging Face (required for gated models)
huggingface-cli login
```

> **Note:** Gemma-2-2B and Llama-3.2-1B both require accepting licence terms on
> the Hugging Face model pages before downloading.

---

## Usage

### Step 1 — Generate attribution graphs

Open and run `notebooks/generate_graphs.ipynb`. This traces both models on all
10 prompts and saves the pruned graphs to `data/graphs/gemma/` and
`data/graphs/llama/`.

### Step 2 — Explore the dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501`. Select a prompt and two models to see the
side-by-side circuit comparison.

### Step 3 — Use the library directly

```python
from src.graph_processor import GraphProcessor
from src.circuit_comparator import CircuitComparator
from src.metrics import compute_feature_overlap

processor = GraphProcessor()
gemma_graph = processor.load_graph("data/graphs/gemma/fact_1.json")
llama_graph  = processor.load_graph("data/graphs/llama/fact_1.json")

comparator = CircuitComparator(gemma_graph, llama_graph)
results = comparator.compare()

print(f"Feature overlap (Jaccard):  {results['metrics']['feature_overlap']['jaccard']:.3f}")
print(f"Structural similarity:      {results['metrics']['structural_similarity']['composite_similarity']:.3f}")
print(f"Layer distribution EMD:     {results['metrics']['layer_distribution']['emd']:.3f}")
print(f"Composite similarity:       {results['metrics']['composite_similarity']:.3f}")

print(comparator.feature_overlap_report())
```

### Running tests

```bash
pytest tests/ -v --tb=short
```

---

## Architecture

```
CircuitLens/
├── notebooks/
│   └── generate_graphs.ipynb    # Run models, save graphs
├── src/
│   ├── graph_processor.py       # Parse circuit-tracer JSON → NetworkX
│   ├── feature_extractor.py     # Pull features, layer profiles, fingerprints
│   ├── circuit_comparator.py    # Orchestrate pairwise comparison
│   ├── metrics.py               # Four similarity metrics
│   └── visualizer.py            # Plotly figures
├── dashboard/
│   ├── app.py                   # Streamlit entry point
│   └── components/
│       ├── circuit_viewer.py    # Single-model card
│       └── comparison_view.py   # Side-by-side comparison UI
├── data/
│   ├── prompts.json             # 10 test prompts
│   └── graphs/gemma|llama/     # Generated attribution graphs (gitignored)
├── results/analysis/            # CSV/JSON outputs (gitignored)
└── tests/
    └── test_metrics.py          # Unit tests for all metrics
```

### Similarity metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Feature Overlap (Jaccard)** | Ratio of shared `(layer, feature_id)` pairs to union | 0–1 |
| **Structural Similarity** | Composite of density, clustering, LCC fraction, degree distribution | 0–1 |
| **Path Statistics** | Average path length and maximum circuit depth | — |
| **Layer Distribution (EMD)** | Earth-mover distance between per-layer activation distributions | ≥0, lower = more similar |

---

## Test prompts

| ID | Prompt | Category |
|----|--------|----------|
| `fact_1` | The capital of France is | factual_recall |
| `math_1` | 2 + 2 = | arithmetic |
| `acronym_1` | The International Space Station (I | acronyms |
| `reasoning_1` | If it's raining… Therefore, the ground is | logical_reasoning |
| `completion_1` | To be or not to be, that is the | completion |
| `fact_2` | Michael Jordan plays the sport of | factual_recall |
| `math_2` | 36 + 59 = | arithmetic |
| `acronym_2` | The National Aeronautics and Space Administration (N | acronyms |
| `reasoning_2` | All birds have wings. A penguin… Therefore, a penguin has | logical_reasoning |
| `multi_step_1` | The capital of the state containing Dallas is | multi_step_reasoning |

---

## Sample findings *(placeholder — populate after running)*

> After generating graphs, update this section with real numbers.

| Category | Gemma Nodes | Llama Nodes | Jaccard | Composite |
|----------|-------------|-------------|---------|-----------|
| factual_recall | — | — | — | — |
| arithmetic | — | — | — | — |
| logical_reasoning | — | — | — | — |

**Preliminary observations:**
- [ ] Factual recall tasks appear to share more circuit structure than arithmetic
- [ ] Llama uses shallower circuits on average for single-hop factual queries
- [ ] Layer distribution diverges most on multi-step reasoning prompts

---

## Extending the project

**Add a new model:** create a folder under `data/graphs/<model_name>/`, generate
graphs with circuit-tracer (set `scan_type` to match the new architecture), and
the dashboard will detect it automatically.

**Add a new metric:** implement a function with signature
`(g1: AttributionGraph, g2: AttributionGraph) -> dict` in `src/metrics.py` and
call it from `compute_all_metrics`.

**Add new prompts:** append entries to `data/prompts.json` and re-run the
notebook.

---

## License

MIT
