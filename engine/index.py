from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dast_schema import CrossReference, DASTNode, NodeType, PhysicalLocation
from .entities import extract_keywords


@dataclass
class DASTIndex:
    root: DASTNode
    nodes_by_id: Dict[str, DASTNode]
    parent_of: Dict[str, Optional[str]]
    children_of: Dict[str, List[str]]
    xref_edges: Dict[str, List[str]]
    term_doc_freq: Dict[str, int]
    total_nodes: int

    def __init__(self, root: DASTNode):
        self.root = root
        self.nodes_by_id = {}
        self.parent_of = {}
        self.children_of = {}
        self.xref_edges = {}
        self._build_indices()
        self.term_doc_freq = self._compute_term_doc_freq()
        self.total_nodes = len(self.nodes_by_id)

    @classmethod
    def from_json(cls, path: str) -> "DASTIndex":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        root = cls._node_from_dict(payload)
        return cls(root)

    @classmethod
    def _node_from_dict(cls, data: Dict[str, Any]) -> DASTNode:
        location = None
        if data.get("physical_location") is not None:
            location = PhysicalLocation(**data["physical_location"])

        node = DASTNode(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            text=data.get("text", "") or "",
            title=data.get("title"),
            level=data.get("level"),
            physical_location=location,
            semantic_path=data.get("semantic_path", []),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {}),
            uuid=data.get("uuid", ""),
            content_hash=data.get("content_hash", ""),
        )
        node.cross_references = [
            CrossReference(
                raw_text=ref["raw_text"],
                target_node_id=ref.get("target_node_id"),
                resolved=ref.get("resolved", False),
            )
            for ref in data.get("cross_references", [])
        ]
        node.children = [cls._node_from_dict(child) for child in data.get("children", [])]
        return node

    def _build_indices(self) -> None:
        for node in self.root.walk():
            self.nodes_by_id[node.node_id] = node
            self.parent_of[node.node_id] = node.parent_id
            self.children_of[node.node_id] = [child.node_id for child in node.children]
            self.xref_edges[node.node_id] = [
                ref.target_node_id for ref in node.cross_references if ref.resolved and ref.target_node_id
            ]

    def _compute_term_doc_freq(self) -> Dict[str, int]:
        df = Counter()
        for node in self.root.walk():
            text = " ".join(filter(None, [node.title or "", node.text or ""]))
            terms = set(extract_keywords(text, stopwords=None))
            for term in terms:
                df[term] += 1
        return dict(df)

    def ancestors(self, node_id: str) -> List[str]:
        parts = node_id.split(".")
        ancestors = []
        for i in range(1, len(parts)):
            ancestors.append(".".join(parts[:i]))
        return ancestors

    def descendants(self, node_id: str) -> List[str]:
        prefix = node_id + "."
        return [nid for nid in self.nodes_by_id if nid.startswith(prefix)]

    def depth(self, node_id: str) -> int:
        return node_id.count(".")

    def incoming_xref_targets(self) -> Dict[str, List[str]]:
        incoming: Dict[str, List[str]] = {}
        for source, targets in self.xref_edges.items():
            for target in targets:
                incoming.setdefault(target, []).append(source)
        return incoming
