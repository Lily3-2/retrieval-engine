"""
PDF -> DAST parser.

PDFs have no explicit structural markup, so this parser must *infer*
structure — this is the hardest and most "real world" of the three parsers.

Strategy:
  1. Use pdfplumber to extract, per page: words with bounding boxes and font
     sizes, and any detected tables (pdfplumber's table-detection algorithm,
     which looks for ruling lines / whitespace grids).
  2. Group words into lines (by y-coordinate proximity), then lines into
     paragraphs (by vertical gap thresholds).
  3. Classify each line as a HEADING candidate if its dominant font size is
     meaningfully larger than the page's median body font size. Assign a
     heading LEVEL by bucketing distinct large font sizes (largest size seen
     = level 1, next distinct size = level 2, etc.) — this is the standard
     heuristic real-world PDF structure extractors use, since PDFs don't
     carry semantic heading tags the way HTML/MD do.
  4. Build the same SECTION-nested tree as the other two parsers, so
     downstream retrieval (Task 2) is format-agnostic.
  5. Physical location carries page_start/page_end AND bbox (PDF is the only
     format where "physical layout" is truly meaningful — this is the
     format the DAST's bbox field exists for).

This is a heuristic layer, not a guarantee — flagged explicitly in the
README as the main source of structural error for PDFs, and a natural
place to plug in a layout model (e.g. LayoutLM) later without touching the
downstream schema.
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from typing import List, Dict, Any
import pdfplumber

from dast_schema import DASTNode, NodeType, PhysicalLocation, next_child_id
from xref_resolver import detect_cross_references, resolve_cross_references

    
def _cluster_lines(words: List[Dict[str, Any]], y_tolerance: float = 3.0):
    """Group words into lines by top-y proximity."""
    lines = []
    for w in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        placed = False
        for line in lines:
            if abs(line["top"] - w["top"]) <= y_tolerance:
                line["words"].append(w)
                line["top"] = min(line["top"], w["top"])
                line["bottom"] = max(line["bottom"], w["bottom"])
                line["x0"] = min(line["x0"], w["x0"])
                line["x1"] = max(line["x1"], w["x1"])
                placed = True
                break
        if not placed:
            lines.append({"top": w["top"], "bottom": w["bottom"],
                          "x0": w["x0"], "x1": w["x1"], "words": [w]})
    lines.sort(key=lambda l: l["top"])
    for line in lines:
        line["words"].sort(key=lambda w: w["x0"])
        line["text"] = " ".join(w["text"] for w in line["words"])
        sizes = [round(w.get("size", 0), 1) for w in line["words"]]
        line["font_size"] = Counter(sizes).most_common(1)[0][0] if sizes else 0
    return lines


class PDFParser:
    def parse(self, path: str, doc_title: str = None) -> DASTNode:
        doc_title = doc_title or os.path.basename(path)
        root = DASTNode(
            node_id="doc",
            node_type=NodeType.DOCUMENT,
            title=doc_title,
            semantic_path=[doc_title],
        )
        child_counters = {"doc": 0}

        def new_child_id(parent: DASTNode) -> str:
            idx = child_counters.get(parent.node_id, 0) + 1
            child_counters[parent.node_id] = idx
            return next_child_id(parent.node_id, idx)

        def attach(parent: DASTNode, node: DASTNode):
            node.parent_id = parent.node_id
            node.semantic_path = parent.semantic_path + ([node.title] if node.title else [])
            parent.children.append(node)
            child_counters[node.node_id] = 0

        all_lines_by_page = []
        body_font_votes = Counter()

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                words = page.extract_words(extra_attrs=["size"])
                lines = _cluster_lines(words)
                all_lines_by_page.append((page_num, page, lines))
                for line in lines:
                    body_font_votes[line["font_size"]] += len(line["text"])

            if not body_font_votes:
                root.finalize()
                return root

            body_size = body_font_votes.most_common(1)[0][0]
            heading_sizes = sorted({s for s in body_font_votes if s > body_size + 0.5}, reverse=True)
            size_to_level = {s: i + 1 for i, s in enumerate(heading_sizes)}

            stack = [(root, 0)]
            tables_consumed = set()  # (page_num, table_index) already emitted

            for page_num, page, lines in all_lines_by_page:
                # extract tables for this page up front, track their bbox for exclusion
                page_tables = page.find_tables()

                def line_in_a_table(line):
                    for t in page_tables:
                        tb = t.bbox  # (x0, top, x1, bottom)
                        if tb[1] - 2 <= line["top"] and line["bottom"] <= tb[3] + 2:
                            return True
                    return False

                for line in lines:
                    if line_in_a_table(line) or not line["text"].strip():
                        continue

                    fsz = line["font_size"]
                    bbox = [round(line["x0"], 1), round(line["top"], 1),
                           round(line["x1"], 1), round(line["bottom"], 1)]
                    loc = PhysicalLocation(source_format="pdf", page_start=page_num,
                                           page_end=page_num, bbox=bbox)

                    if fsz in size_to_level:
                        level = size_to_level[fsz]
                        while len(stack) > 1 and stack[-1][1] >= level:
                            stack.pop()
                        parent = stack[-1][0]
                        sec = DASTNode(node_id=new_child_id(parent), node_type=NodeType.SECTION,
                                       title=line["text"], level=level, physical_location=loc)
                        sec.cross_references = detect_cross_references(line["text"])
                        attach(parent, sec)
                        stack.append((sec, level))
                    else:
                        parent = stack[-1][0]
                        node = DASTNode(node_id=new_child_id(parent), node_type=NodeType.PARAGRAPH,
                                        text=line["text"], physical_location=loc)
                        node.cross_references = detect_cross_references(line["text"])
                        attach(parent, node)

                # tables: attach to whatever section is currently open on this page
                for t_idx, table in enumerate(page_tables):
                    key = (page_num, t_idx)
                    if key in tables_consumed:
                        continue
                    tables_consumed.add(key)
                    data = table.extract()
                    if not data:
                        continue
                    parent = stack[-1][0]
                    tb = [round(v, 1) for v in table.bbox]
                    table_node = DASTNode(node_id=new_child_id(parent), node_type=NodeType.TABLE,
                                          physical_location=PhysicalLocation(
                                              source_format="pdf", page_start=page_num,
                                              page_end=page_num, bbox=tb))
                    attach(parent, table_node)
                    for r_idx, row in enumerate(data):
                        row_node = DASTNode(node_id=new_child_id(table_node),
                                            node_type=NodeType.TABLE_ROW, level=r_idx)
                        attach(table_node, row_node)
                        for cell_val in row:
                            cell_node = DASTNode(node_id=new_child_id(row_node),
                                                 node_type=NodeType.TABLE_CELL,
                                                 text=(cell_val or "").strip())
                            attach(row_node, cell_node)

 
        resolve_cross_references(root)
        root.finalize()
        return root
    
if __name__ == "__main__":
    pdf_path = "D:/Mine/Post-grad-prep/retrieval-engine/benchmark/document.pdf"

    parser = PDFParser()
    root = parser.parse(pdf_path)

    print(f"Parsed document: {root.title}")