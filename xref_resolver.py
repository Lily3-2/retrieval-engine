"""
Cross-reference detection and resolution.

Detection: regex over node text for common technical-doc reference patterns
("Section 2.1", "Table 3", "Figure 4", "see Appendix B", "as per Clause 5").

Resolution: a second pass, run after the whole tree is built, that matches
each raw reference string against section/figure/table TITLES elsewhere in
the tree and fills in target_node_id. This is what makes cross-references a
real graph edge instead of just a string — Task 2's traversal engine can
follow these edges directly.
"""

import re
from typing import List
from dast_schema import DASTNode, CrossReference, NodeType

REF_PATTERNS = [
    re.compile(r"\b(Section|Sec\.?)\s+([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE),
    re.compile(r"\b(Table)\s+([0-9]+)", re.IGNORECASE),
    re.compile(r"\b(Figure|Fig\.?)\s+([0-9]+)", re.IGNORECASE),
    re.compile(r"\b(Appendix)\s+([A-Z0-9]+)", re.IGNORECASE),
    re.compile(r"\b(Clause)\s+([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE),
]


def detect_cross_references(text: str) -> List[CrossReference]:
    refs = []
    seen = set()
    for pattern in REF_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(0)
            if raw.lower() not in seen:
                seen.add(raw.lower())
                refs.append(CrossReference(raw_text=raw))
    return refs


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def resolve_cross_references(root: DASTNode) -> None:
    """Second pass: match raw_text against titles of SECTION/FIGURE/TABLE nodes."""
    title_index = {}
    for n in root.walk():
        if n.title:
            title_index[_normalize(n.title)] = n.node_id
        # also index by leading label e.g. "Table 3" if title starts with it
        if n.node_type in (NodeType.TABLE, NodeType.FIGURE) and n.title:
            title_index[_normalize(n.title)] = n.node_id

    for n in root.walk():
        for ref in n.cross_references:
            key_full = _normalize(ref.raw_text)
            if key_full in title_index:
                ref.target_node_id = title_index[key_full]
                ref.resolved = True
                continue
            # loose match: does any title start with the ref text?
            for title_norm, node_id in title_index.items():
                if title_norm.startswith(key_full) or key_full in title_norm:
                    ref.target_node_id = node_id
                    ref.resolved = True
                    break
