from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from dast_schema import DASTNode
from .entities import Query, build_query
from .index import DASTIndex
from .scoring import (
    DEFAULT_WEIGHTS,
    ancestor_descendant_score,
    entity_score,
    keyword_score,
    path_score,
    structural_score,
    xref_score,
)
from .traversal import bfs, dfs, best_first


@dataclass
class ScoredNode:
    node_id: str
    score: float
    factors: Dict[str, float]
    matched_terms: List[str]
    semantic_path: List[str]
    physical_location: Dict[str, Any]
    text: str


ScoreFn = Callable[[Query, DASTNode], ScoredNode]


def _matched_terms(query: Query, node: DASTNode) -> List[str]:
    source = " ".join(filter(None, [node.title or "", node.text or ""]))
    lowered = source.lower()
    return [term for term in query.keywords if term in lowered]


def _score_node(q: Query, node: DASTNode, index: DASTIndex, weights: Dict[str, float]) -> ScoredNode:
    factors = {
        "keyword": keyword_score(q, node, index),
        "structural": structural_score(q, node, index),
        "entity": entity_score(q, node, index),
        "xref": xref_score(q, node, index),
        "ancestor": ancestor_descendant_score(q, node, index),
        "path": path_score(q, node, index),
    }
    total = sum(weights.get(name, 0.0) * value for name, value in factors.items())
    score = float(total)
    return ScoredNode(
        node_id=node.node_id,
        score=score,
        factors=factors,
        matched_terms=_matched_terms(q, node),
        semantic_path=node.semantic_path,
        physical_location=node.physical_location.to_dict() if node.physical_location else {},
        text=node.text or node.title or "",
    )


class DASTRetriever:
    def __init__(self, index: DASTIndex, weights: Dict[str, float] = None, strategy: str = "best_first", beam: int | None = None, prune_threshold: float = 0.0):
        self.index = index
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS
        self.strategy = strategy
        self.beam = beam
        self.prune_threshold = prune_threshold
        self._strategy_map = {
            "best_first": best_first,
            "bfs": bfs,
            "dfs": dfs,
        }

    def _score_fn(self, q: Query) -> ScoreFn:
        return lambda query, node: _score_node(query, node, self.index, self.weights)

    def retrieve(self, question: str, top_k: int = 8) -> List[ScoredNode]:
        query = build_query(question)
        score_fn = self._score_fn(query)
        strategy_fn = self._strategy_map.get(self.strategy)
        if strategy_fn is None:
            raise ValueError(f"Unsupported traversal strategy: {self.strategy}")
        results = strategy_fn(self.index, query, score_fn, top_k, beam=self.beam, prune_threshold=self.prune_threshold)
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]
