"""
Markdown -> DAST parser.

Strategy:
  1. Tokenize the markdown into block-level tokens using `markdown_it` (a
     CommonMark-compliant tokenizer) so we get real structural tokens
     (heading, paragraph, list, table, fence/code, blockquote) with source
     line maps, instead of regex-hacking the raw text.
  2. Walk the token stream and build a SECTION-nested tree: every heading
     opens a new SECTION node at the appropriate depth (closing any open
     sections at >= its level); everything else becomes a child of the
     current innermost open section.
  3. Attach line_start/line_end (from markdown-it's `.map`) as physical
     location, and char offsets computed from line boundaries.
  4. Run cross-reference detection on every node's text.
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional
from markdown_it import MarkdownIt

from dast_schema import DASTNode, NodeType, PhysicalLocation, next_child_id
from xref_resolver import detect_cross_references, resolve_cross_references


def _line_char_offsets(source: str) -> List[int]:
    """offsets[i] = char index where line i (0-based) starts."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


class MarkdownParser:
    def __init__(self):
        self.md = MarkdownIt("commonmark", {"html": False}).enable("table")

    def parse(self, source: str, doc_title: str = "Document") -> DASTNode:
        tokens = self.md.parse(source)
        line_offsets = _line_char_offsets(source)

        root = DASTNode(
            node_id="doc",
            node_type=NodeType.DOCUMENT,
            title=doc_title,
            semantic_path=[doc_title],
            physical_location=PhysicalLocation(
                source_format="markdown", line_start=0, line_end=source.count("\n")
            ),
        )

        # stack of (section_node, heading_level); root counts as level 0
        stack = [(root, 0)]
        child_counters = {"doc": 0}  # node_id -> next child index

        def new_child_id(parent: DASTNode) -> str:
            idx = child_counters.get(parent.node_id, 0) + 1
            child_counters[parent.node_id] = idx
            return next_child_id(parent.node_id, idx)

        def attach(parent: DASTNode, node: DASTNode):
            node.parent_id = parent.node_id
            node.semantic_path = parent.semantic_path + ([node.title] if node.title else [])
            parent.children.append(node)

        def loc_for_map(map_range) -> Optional[PhysicalLocation]:
            if not map_range:
                return None
            ls, le = map_range[0], map_range[1]
            cs = line_offsets[ls] if ls < len(line_offsets) else None
            ce = line_offsets[le] if le < len(line_offsets) else None
            return PhysicalLocation(
                source_format="markdown", line_start=ls, line_end=le,
                char_start=cs, char_end=ce,
            )

        i = 0
        n = len(tokens)
        while i < n:
            tok = tokens[i]

            if tok.type == "heading_open":
                level = int(tok.tag[1])  # h1 -> 1, h2 -> 2, ...
                inline_tok = tokens[i + 1]
                title_text = inline_tok.content
                # close sections at same or deeper level
                while len(stack) > 1 and stack[-1][1] >= level:
                    stack.pop()
                parent = stack[-1][0]
                sec = DASTNode(
                    node_id=new_child_id(parent),
                    node_type=NodeType.SECTION,
                    title=title_text,
                    level=level,
                    physical_location=loc_for_map(tok.map),
                )
                attach(parent, sec)
                sec.cross_references = detect_cross_references(title_text)
                stack.append((sec, level))
                child_counters[sec.node_id] = 0
                i += 3  # heading_open, inline, heading_close
                continue

            if tok.type == "paragraph_open":
                inline_tok = tokens[i + 1]
                text = inline_tok.content
                parent = stack[-1][0]
                node = DASTNode(
                    node_id=new_child_id(parent),
                    node_type=NodeType.PARAGRAPH,
                    text=text,
                    physical_location=loc_for_map(tok.map),
                )
                node.cross_references = detect_cross_references(text)
                attach(parent, node)
                i += 3
                continue

            if tok.type == "bullet_list_open" or tok.type == "ordered_list_open":
                ordered = tok.type == "ordered_list_open"
                parent = stack[-1][0]
                list_node = DASTNode(
                    node_id=new_child_id(parent),
                    node_type=NodeType.LIST,
                    physical_location=loc_for_map(tok.map),
                    metadata={"ordered": ordered},
                )
                attach(parent, list_node)
                child_counters[list_node.node_id] = 0
                i = self._consume_list_items(tokens, i + 1, list_node, new_child_id,
                                             attach, loc_for_map, child_counters)
                continue

            if tok.type == "table_open":
                parent = stack[-1][0]
                table_node = DASTNode(
                    node_id=new_child_id(parent),
                    node_type=NodeType.TABLE,
                    physical_location=loc_for_map(tok.map),
                )
                attach(parent, table_node)
                child_counters[table_node.node_id] = 0
                i = self._consume_table(tokens, i + 1, table_node, new_child_id,
                                        attach, loc_for_map, child_counters)
                continue
            
            if tok.type == "fence" or tok.type == "code_block":
                parent = stack[-1][0]
                node = DASTNode(
                    node_id=new_child_id(parent),
                    node_type=NodeType.CODE_BLOCK,
                    text=tok.content,
                    metadata={"language": tok.info.strip() if tok.info else None},
                    physical_location=loc_for_map(tok.map),
                )
                attach(parent, node)
                i += 1
                continue

            if tok.type == "blockquote_open":
                parent = stack[-1][0]
                # gather inline text of paragraphs until blockquote_close
                j = i + 1
                texts = []
                start_map = tok.map
                while j < n and tokens[j].type != "blockquote_close":
                    if tokens[j].type == "inline":
                        texts.append(tokens[j].content)
                    j += 1
                node = DASTNode(
                    node_id=new_child_id(parent),
                    node_type=NodeType.QUOTE,
                    text="\n".join(texts),
                    physical_location=loc_for_map(start_map),
                )
                node.cross_references = detect_cross_references(node.text)
                attach(parent, node)
                i = j + 1
                continue

            if tok.type == "hr":
                parent = stack[-1][0]
                node = DASTNode(
                    node_id=new_child_id(parent),
                    node_type=NodeType.THEMATIC_BREAK,
                    physical_location=loc_for_map(tok.map),
                )
                attach(parent, node)
                i += 1
                continue

            i += 1

        resolve_cross_references(root)
        root.finalize()
        return root

    def _consume_list_items(self, tokens, i, list_node, new_child_id, attach,
                            loc_for_map, child_counters):
        n = len(tokens)
        while i < n and tokens[i].type != "bullet_list_close" and tokens[i].type != "ordered_list_close":
            if tokens[i].type == "list_item_open":
                item_map = tokens[i].map
                j = i + 1
                texts = []
                while j < n and tokens[j].type != "list_item_close":
                    if tokens[j].type == "inline":
                        texts.append(tokens[j].content)
                    j += 1
                item = DASTNode(
                    node_id=new_child_id(list_node),
                    node_type=NodeType.LIST_ITEM,
                    text=" ".join(texts),
                    physical_location=loc_for_map(item_map),
                )
                item.cross_references = detect_cross_references(item.text)
                attach(list_node, item)
                i = j + 1
            else:
                i += 1
        return i + 1  # skip the closing token

    def _consume_table(self, tokens, i, table_node, new_child_id, attach,
                       loc_for_map, child_counters):
        n = len(tokens)
        row_idx = 0
        while i < n and tokens[i].type != "table_close":
            if tokens[i].type in ("tr_open",):
                row_map = tokens[i].map
                row_node = DASTNode(
                    node_id=new_child_id(table_node),
                    node_type=NodeType.TABLE_ROW,
                    level=row_idx,
                    physical_location=loc_for_map(row_map),
                )
                attach(table_node, row_node)
                child_counters[row_node.node_id] = 0
                row_idx += 1
                j = i + 1
                while j < n and tokens[j].type != "tr_close":
                    if tokens[j].type in ("th_open", "td_open"):
                        is_header = tokens[j].type == "th_open"
                        inline_tok = tokens[j + 1]
                        cell = DASTNode(
                            node_id=new_child_id(row_node),
                            node_type=NodeType.TABLE_CELL,
                            text=inline_tok.content,
                            metadata={"is_header": is_header},
                        )
                        attach(row_node, cell)
                        j += 3
                    else:
                        j += 1
                i = j + 1
            else:
                i += 1
        return i + 1
if __name__ == "__main__":
    md_file = "/Users/anushkaupadhyay/Downloads/dast_poc/demo/sample.md"

    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    parser = MarkdownParser()
    root = parser.parse(
        md_content,
        doc_title=os.path.basename(md_file)
    )
    print(f"Parsed document: {root.title}")
    output_dir = "/Users/anushkaupadhyay/Downloads/dast_poc/output"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "Parser_md.dast.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(root.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"DAST saved to {output_file}")