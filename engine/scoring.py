from __future__ import annotations

import math
from collections import Counter
from typing import Dict

from dast_schema import DASTNode
from .entities import Query, extract_entities, extract_keywords

DEFAULT_WEIGHTS = {
    "keyword": 0.35,
    "structural": 0.10,
    "entity": 0.20,
    "xref": 0.05,
    "ancestor": 0.15,
    "path": 0.15,
}

NODE_TYPE_WEIGHT = {
    "document": 0.2,
    "section": 1.0,
    "paragraph": 0.7,
    "list": 0.5,
    "list_item": 0.6,
    "table": 0.9,
    "table_row": 0.4,
    "table_cell": 0.3,
    "code_block": 0.6,
    "figure": 0.5,
    "caption": 0.4,
    "quote": 0.5,
    "reference": 0.4,
    "footnote": 0.2,
    "thematic_break": 0.1,
}


def _node_term_counts(node: DASTNode) -> Counter[str]:
    text = " ".join(filter(None, [node.title or "", node.text or ""]))
    terms = extract_keywords(text)
    return Counter(terms)


def _overlap_ratio(query_terms: list[str], target_terms: list[str]) -> float:
    if not query_terms or not target_terms:
        return 0.0
    matches = sum(1 for term in query_terms if term in target_terms)
    return min(1.0, matches / len(query_terms))


def keyword_score(q: Query, node: DASTNode, index) -> float:
    query_counts = Counter(q.keywords)
    node_counts = _node_term_counts(node)
    if not query_counts or not node_counts:
        return 0.0
    score = 0.0
    for term, freq in query_counts.items():
        tf = min(node_counts.get(term, 0), 3)
        df = index.term_doc_freq.get(term, 0)
        idf = math.log((index.total_nodes + 1) / (df + 1)) + 1.0
        score += tf * idf
    normalized = score / (sum(node_counts.values()) + 1)
    return min(1.0, normalized)


def structural_score(q: Query, node: DASTNode, index) -> float:
    base = NODE_TYPE_WEIGHT.get(node.node_type.value, 0.2)
    depth = index.depth(node.node_id)
    depth_boost = 0.1 if depth <= 2 else 0.0
    return min(1.0, base + depth_boost)


def entity_score(q: Query, node: DASTNode, index) -> float:
    if not q.entities:
        return 0.0
    node_entities = extract_entities(" ".join(filter(None, [node.title or "", node.text or ""])))
    if not node_entities:
        return 0.0
    matches = {e.lower() for e in node_entities} & {e.lower() for e in q.entities}
    return min(1.0, len(matches) / max(1, len(q.entities)))


def xref_score(q: Query, node: DASTNode, index) -> float:
    incoming = index.incoming_xref_targets()
    if node.node_id in incoming:
        return 1.0
    query_labels = {ref.lower() for ref in q.raw_refs}
    title = (node.title or "").lower()
    if any(label in title for label in query_labels):
        return 1.0
    return 0.0


def ancestor_descendant_score(q: Query, node: DASTNode, index) -> float:
    ancestor_terms = []
    for ancestor_id in index.ancestors(node.node_id):
        ancestor = index.nodes_by_id.get(ancestor_id)
        if ancestor and ancestor.title:
            ancestor_terms.extend(extract_keywords(ancestor.title))
    child_terms = []
    for child_id in index.children_of.get(node.node_id, []):
        child = index.nodes_by_id.get(child_id)
        if child:
            child_terms.extend(extract_keywords(" ".join(filter(None, [child.title or "", child.text or ""]))))
    ratio_anc = _overlap_ratio(q.keywords, ancestor_terms)
    ratio_child = _overlap_ratio(q.keywords, child_terms)
    return min(1.0, max(ratio_anc, ratio_child))


def path_score(q: Query, node: DASTNode, index) -> float:
    path_terms = []
    for fragment in node.semantic_path:
        path_terms.extend(extract_keywords(fragment))
    return min(1.0, _overlap_ratio(q.keywords, path_terms))
