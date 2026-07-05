"""
DAST — Document Abstract Syntax Tree
======================================
Core schema for representing a structured technical document as an executable
tree of semantic nodes, rather than as unstructured text.

Design answers to the three research questions for Task 1:

1) WHAT SEMANTIC NODE TYPES ARE NEEDED?
   A closed set of NodeType values (below) covering the structural primitives
   that recur across technical documents (specs, manuals, papers, contracts,
   reports): sectioning, prose, lists, tables, code, figures, quotes, and
   references. Every parser (md/html/pdf) must normalize into this same set,
   which is what makes retrieval format-agnostic.

2) HOW SHOULD PHYSICAL LOCATION BE REPRESENTED?
   A PhysicalLocation object attached to every node, carrying whichever of
   these apply to the source format:
     - page_start/page_end   (PDF, and HTML/MD if paginated upstream)
     - bbox                  (PDF: x0,y0,x1,y1 in PDF points)
     - line_start/line_end   (Markdown/HTML source lines)
     - char_start/char_end   (offsets into the raw source string)
   This lets us jump from a retrieved node straight back to "page 12,
   top-right box" or "source line 340-355" — the thing embeddings can't do.

3) HOW DO WE UNIQUELY IDENTIFY EVERY NODE?
   Three complementary identifiers, each serving a different purpose:
     - node_id       : deterministic, human-readable, position-encoding path
                        e.g. "doc.2.1.3" = 3rd child of 1st child of 2nd
                        top-level section. Encodes parent/order for free.
     - uuid           : random UUID4, stable identity independent of any
                        future re-ordering/re-parsing of the document.
     - content_hash   : sha1 of (semantic_path + normalized text). Lets you
                        detect "this exact content already existed under
                        this heading" across re-parses/versions — a cheap
                        way to diff document revisions without re-embedding.
   node_id is what retrieval/traversal algorithms use (Task 2); content_hash
   is what change-detection / caching (mentioned in "cosmetics") would use.
"""

from __future__ import annotations

import hashlib
import uuid as uuid_lib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any


class NodeType(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"          # heading + everything under it until the next same/higher heading
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    CODE_BLOCK = "code_block"
    FIGURE = "figure"
    CAPTION = "caption"
    QUOTE = "quote"
    REFERENCE = "reference"      # an inline cross-reference target/anchor
    FOOTNOTE = "footnote"
    THEMATIC_BREAK = "thematic_break"


@dataclass
class PhysicalLocation:
    source_format: str                      # "markdown" | "html" | "pdf"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    bbox: Optional[List[float]] = None      # [x0, y0, x1, y1], PDF points
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class CrossReference:
    """A reference found inside a node's text pointing elsewhere in the doc."""
    raw_text: str                            # e.g. "see Section 2.1", "Table 3"
    target_node_id: Optional[str] = None     # resolved after full tree is built
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "target_node_id": self.target_node_id,
            "resolved": self.resolved,
        }


@dataclass
class DASTNode:
    node_id: str
    node_type: NodeType
    text: str = ""
    title: Optional[str] = None
    level: Optional[int] = None              # heading depth, list nesting depth, etc.
    physical_location: Optional[PhysicalLocation] = None
    semantic_path: List[str] = field(default_factory=list)   # breadcrumb of ancestor titles
    parent_id: Optional[str] = None
    children: List["DASTNode"] = field(default_factory=list)
    cross_references: List[CrossReference] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    uuid: str = field(default_factory=lambda: str(uuid_lib.uuid4()))
    content_hash: str = ""

    def compute_content_hash(self) -> str:
        basis = "|".join(self.semantic_path + [self.node_type.value, self.text.strip()])
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]

    def finalize(self) -> None:
        """Call after text/semantic_path are set, before/while serializing."""
        self.content_hash = self.compute_content_hash()
        for c in self.children:
            c.finalize()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "uuid": self.uuid,
            "content_hash": self.content_hash,
            "node_type": self.node_type.value,
            "title": self.title,
            "text": self.text,
            "level": self.level,
            "semantic_path": self.semantic_path,
            "parent_id": self.parent_id,
            "physical_location": self.physical_location.to_dict() if self.physical_location else None,
            "cross_references": [r.to_dict() for r in self.cross_references],
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }

    # --- convenience traversal helpers, used by both parsers and Task 2 ---

    def walk(self):
        """Depth-first generator over this node and all descendants."""
        yield self
        for c in self.children:
            yield from c.walk()

    def find_by_id(self, node_id: str) -> Optional["DASTNode"]:
        for n in self.walk():
            if n.node_id == node_id:
                return n
        return None


def next_child_id(parent_id: str, index: int) -> str:
    """Deterministic path-based ID: parent path + '.' + positional index."""
    return f"{parent_id}.{index}"
