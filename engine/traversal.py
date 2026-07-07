from __future__ import annotations

import heapq
from collections import deque
from typing import Callable, List

from .types import ScoredNode
from dast_schema import DASTNode

ScoreFn = Callable[["Query", DASTNode], ScoredNode]


def bfs(index, q, score_fn: ScoreFn, top_k: int, beam: int | None = None, prune_threshold: float = 0.0) -> List[ScoredNode]:
    queue = deque([index.root])
    scored = []
    while queue:
        node = queue.popleft()
        scored.append(score_fn(q, node))
        for child in node.children:
            queue.append(child)
    return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def dfs(index, q, score_fn: ScoreFn, top_k: int, beam: int | None = None, prune_threshold: float = 0.0) -> List[ScoredNode]:
    scored = []

    def _visit(node: DASTNode) -> None:
        result = score_fn(q, node)
        scored.append(result)
        if node.node_type.value == "section" and result.score < prune_threshold:
            return
        for child in node.children:
            _visit(child)

    _visit(index.root)
    return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def best_first(index, q, score_fn: ScoreFn, top_k: int, beam: int | None = None, prune_threshold: float = 0.0) -> List[ScoredNode]:
    heap = []
    counter = 0
    initial = score_fn(q, index.root)
    heapq.heappush(heap, (-initial.score, counter, index.root, initial))
    scored = {}

    while heap and len(scored) < max(top_k, 512):
        _, _, node, node_score = heapq.heappop(heap)
        if node.node_id in scored:
            continue
        scored[node.node_id] = node_score
        if node.node_type.value == "section" and node_score.score < prune_threshold:
            continue
        for child in node.children:
            counter += 1
            child_score = score_fn(q, child)
            heapq.heappush(heap, (-child_score.score, counter, child, child_score))
        if beam is not None and len(heap) > beam:
            heap = heapq.nsmallest(beam, heap)
            heapq.heapify(heap)

    results = sorted(scored.values(), key=lambda item: item.score, reverse=True)
    return results[:top_k]
