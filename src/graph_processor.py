"""Parse circuit-tracer attribution graph JSON files into NetworkX graphs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NodeInfo:
    node_id: str
    node_type: str          # "feature" | "token" | "logit"
    layer: int | None
    position: int | None
    feature_id: int | None
    activation: float
    label: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeInfo:
    source: str
    target: str
    weight: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributionGraph:
    """Parsed attribution graph with convenient accessors."""

    graph: nx.DiGraph
    model_name: str
    prompt_id: str
    prompt_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- convenience properties ----------------------------------------

    @property
    def nodes(self) -> list[NodeInfo]:
        return [d["info"] for _, d in self.graph.nodes(data=True)]

    @property
    def edges(self) -> list[EdgeInfo]:
        return [d["info"] for _, _, d in self.graph.edges(data=True)]

    @property
    def feature_nodes(self) -> list[NodeInfo]:
        return [n for n in self.nodes if n.node_type == "feature"]

    @property
    def token_nodes(self) -> list[NodeInfo]:
        return [n for n in self.nodes if n.node_type == "token"]

    @property
    def logit_nodes(self) -> list[NodeInfo]:
        return [n for n in self.nodes if n.node_type == "logit"]

    @property
    def num_layers(self) -> int:
        layers = {n.layer for n in self.feature_nodes if n.layer is not None}
        return max(layers) + 1 if layers else 0

    def nodes_in_layer(self, layer: int) -> list[NodeInfo]:
        return [n for n in self.feature_nodes if n.layer == layer]

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "prompt_id": self.prompt_id,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_layers": self.num_layers,
            "num_features": len(self.feature_nodes),
            "num_tokens": len(self.token_nodes),
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class GraphProcessor:
    """Load and parse circuit-tracer attribution graph JSON files.

    circuit-tracer emits a JSON structure with the following top-level keys:
        - "nodes": list of node objects
        - "edges": list of edge objects
        - "metadata": optional dict with prompt, model info, etc.

    Node object fields (all optional except "node_id"):
        node_id, type, layer, position, feature_id, activation, label

    Edge object fields:
        source, target, weight
    """

    # Keys circuit-tracer uses — adjust if the library changes its schema.
    _NODE_ID_KEYS = ("node_id", "id", "name")
    _NODE_TYPE_KEYS = ("type", "node_type", "kind")
    _LAYER_KEYS = ("layer", "layer_idx")
    _POSITION_KEYS = ("position", "pos", "token_position")
    _FEATURE_KEYS = ("feature_id", "feature", "feat_id")
    _ACTIVATION_KEYS = ("activation", "act", "value", "weight")
    _LABEL_KEYS = ("label", "name", "description")

    _EDGE_SRC_KEYS = ("source", "src", "from")
    _EDGE_TGT_KEYS = ("target", "tgt", "to", "dst")
    _EDGE_W_KEYS = ("weight", "w", "effect", "direct_effect")

    def load_graph(
        self,
        path: str | Path,
        model_name: str = "",
        prompt_id: str = "",
        prompt_text: str = "",
    ) -> AttributionGraph:
        """Load a single attribution graph JSON file."""
        path = Path(path)
        with path.open() as f:
            raw = json.load(f)

        # Infer identifiers from path if not provided.
        if not prompt_id:
            prompt_id = path.stem
        if not model_name:
            # Expect .../graphs/<model_name>/<prompt_id>.json
            model_name = path.parent.name

        metadata = raw.get("metadata", {})
        if not prompt_text:
            prompt_text = metadata.get("prompt", metadata.get("prompt_text", ""))

        raw_nodes: list[dict] = raw.get("nodes", [])
        raw_edges: list[dict] = raw.get("edges", [])

        G = nx.DiGraph()
        self._add_nodes(G, raw_nodes)
        self._add_edges(G, raw_edges)

        return AttributionGraph(
            graph=G,
            model_name=model_name,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            metadata=metadata,
        )

    def load_graphs_for_prompt(
        self,
        prompt_id: str,
        graphs_dir: str | Path = "data/graphs",
    ) -> dict[str, AttributionGraph]:
        """Return {model_name: AttributionGraph} for all models that have a file for prompt_id."""
        graphs_dir = Path(graphs_dir)
        result: dict[str, AttributionGraph] = {}
        for model_dir in sorted(graphs_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            candidate = model_dir / f"{prompt_id}.json"
            if candidate.exists():
                result[model_dir.name] = self.load_graph(candidate)
        return result

    # ---- private helpers --------------------------------------------------

    def _pick(self, d: dict, keys: tuple[str, ...], default: Any = None) -> Any:
        for k in keys:
            if k in d:
                return d[k]
        return default

    def _add_nodes(self, G: nx.DiGraph, raw_nodes: list[dict]) -> None:
        for raw in raw_nodes:
            node_id = str(self._pick(raw, self._NODE_ID_KEYS, id(raw)))
            node_type = str(self._pick(raw, self._NODE_TYPE_KEYS, "feature")).lower()
            layer_raw = self._pick(raw, self._LAYER_KEYS)
            layer = int(layer_raw) if layer_raw is not None else None
            position_raw = self._pick(raw, self._POSITION_KEYS)
            position = int(position_raw) if position_raw is not None else None
            feature_id_raw = self._pick(raw, self._FEATURE_KEYS)
            feature_id = int(feature_id_raw) if feature_id_raw is not None else None
            activation = float(self._pick(raw, self._ACTIVATION_KEYS, 0.0))
            label = str(self._pick(raw, self._LABEL_KEYS, node_id))

            info = NodeInfo(
                node_id=node_id,
                node_type=node_type,
                layer=layer,
                position=position,
                feature_id=feature_id,
                activation=activation,
                label=label,
                raw=raw,
            )
            G.add_node(node_id, info=info)

    def _add_edges(self, G: nx.DiGraph, raw_edges: list[dict]) -> None:
        for raw in raw_edges:
            src = str(self._pick(raw, self._EDGE_SRC_KEYS, ""))
            tgt = str(self._pick(raw, self._EDGE_TGT_KEYS, ""))
            weight = float(self._pick(raw, self._EDGE_W_KEYS, 0.0))
            if not src or not tgt:
                continue
            # Add placeholder nodes for any dangling references.
            for nid in (src, tgt):
                if nid not in G:
                    G.add_node(nid, info=NodeInfo(node_id=nid, node_type="unknown",
                                                   layer=None, position=None,
                                                   feature_id=None, activation=0.0))
            info = EdgeInfo(source=src, target=tgt, weight=weight, raw=raw)
            G.add_edge(src, tgt, info=info, weight=abs(weight))
