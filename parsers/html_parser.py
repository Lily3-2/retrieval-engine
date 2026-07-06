"""
HTML -> DAST parser.

Strategy: walk the parsed DOM (BeautifulSoup) depth-first. h1-h6 open/close
SECTION nodes exactly like in the markdown parser (same nesting-by-level
logic), so the two parsers converge on an identical tree shape for
equivalent content. Physical location here is char offsets into the raw
HTML source (line numbers are unreliable in HTML since browsers/authors
don't respect them); page numbers are omitted (not applicable to HTML).
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from bs4 import BeautifulSoup, Tag, NavigableString

from dast_schema import DASTNode, NodeType, PhysicalLocation, next_child_id
from xref_resolver import detect_cross_references, resolve_cross_references

HEADING_RE = re.compile(r"^h([1-6])$")


class HTMLParser:
    def parse(self, source: str, doc_title: str = "Document") -> DASTNode:
        soup = BeautifulSoup(source, "lxml")

        root = DASTNode(
            node_id="doc",
            node_type=NodeType.DOCUMENT,
            title=doc_title,
            semantic_path=[doc_title],
            physical_location=PhysicalLocation(source_format="html",
                                                char_start=0, char_end=len(source)),
        )

        stack = [(root, 0)]
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

        def offsets(tag: Tag):
            # BeautifulSoup + lxml doesn't give byte offsets natively;
            # approximate using sourceline if available (lxml exposes it).
            line = getattr(tag, "sourceline", None)
            return PhysicalLocation(source_format="html", line_start=line, line_end=line)

        body = soup.body or soup

        def walk(node_tag):
            for child in node_tag.children:
                if isinstance(child, NavigableString):
                    continue
                if not isinstance(child, Tag):
                    continue
                name = child.name.lower() if child.name else ""
                m = HEADING_RE.match(name)

                if m:
                    level = int(m.group(1))
                    title_text = child.get_text(strip=True)
                    while len(stack) > 1 and stack[-1][1] >= level:
                        stack.pop()
                    parent = stack[-1][0]
                    sec = DASTNode(
                        node_id=new_child_id(parent),
                        node_type=NodeType.SECTION,
                        title=title_text,
                        level=level,
                        physical_location=offsets(child),
                    )
                    sec.cross_references = detect_cross_references(title_text)
                    attach(parent, sec)
                    stack.append((sec, level))
                    continue

                parent = stack[-1][0]

                if name == "p":
                    text = child.get_text(strip=True)
                    if not text:
                        continue
                    node = DASTNode(node_id=new_child_id(parent), node_type=NodeType.PARAGRAPH,
                                     text=text, physical_location=offsets(child))
                    node.cross_references = detect_cross_references(text)
                    attach(parent, node)

                elif name in ("ul", "ol"):
                    list_node = DASTNode(node_id=new_child_id(parent), node_type=NodeType.LIST,
                                          physical_location=offsets(child),
                                          metadata={"ordered": name == "ol"})
                    attach(parent, list_node)
                    for li in child.find_all("li", recursive=False):
                        text = li.get_text(strip=True)
                        item = DASTNode(node_id=new_child_id(list_node), node_type=NodeType.LIST_ITEM,
                                         text=text, physical_location=offsets(li))
                        item.cross_references = detect_cross_references(text)
                        attach(list_node, item)

                elif name == "table":
                    table_node = DASTNode(node_id=new_child_id(parent), node_type=NodeType.TABLE,
                                           physical_location=offsets(child))
                    attach(parent, table_node)
                    rows = child.find_all("tr")
                    for r_idx, tr in enumerate(rows):
                        row_node = DASTNode(node_id=new_child_id(table_node),
                                             node_type=NodeType.TABLE_ROW, level=r_idx,
                                             physical_location=offsets(tr))
                        attach(table_node, row_node)
                        for cell in tr.find_all(["td", "th"]):
                            is_header = cell.name == "th"
                            cell_node = DASTNode(node_id=new_child_id(row_node),
                                                  node_type=NodeType.TABLE_CELL,
                                                  text=cell.get_text(strip=True),
                                                  metadata={"is_header": is_header})
                            attach(row_node, cell_node)

                elif name in ("pre", "code"):
                    node = DASTNode(node_id=new_child_id(parent), node_type=NodeType.CODE_BLOCK,
                                     text=child.get_text(),
                                     physical_location=offsets(child))
                    attach(parent, node)

                elif name == "blockquote":
                    text = child.get_text(strip=True)
                    node = DASTNode(node_id=new_child_id(parent), node_type=NodeType.QUOTE,
                                     text=text, physical_location=offsets(child))
                    node.cross_references = detect_cross_references(text)
                    attach(parent, node)

                elif name in ("figure", "img"):
                    caption = ""
                    figcap = child.find("figcaption") if name == "figure" else None
                    if figcap:
                        caption = figcap.get_text(strip=True)
                    node = DASTNode(node_id=new_child_id(parent), node_type=NodeType.FIGURE,
                                     title=caption or child.get("alt", ""),
                                     physical_location=offsets(child),
                                     metadata={"src": child.get("src") if name == "img" else None})
                    attach(parent, node)

                elif name == "hr":
                    node = DASTNode(node_id=new_child_id(parent), node_type=NodeType.THEMATIC_BREAK,
                                     physical_location=offsets(child))
                    attach(parent, node)

                else:
                    # container tags (div, section, article, span, etc.) - recurse into them
                    walk(child)

        walk(body)
        resolve_cross_references(root)
        root.finalize()
        output_dir = "/Users/anushkaupadhyay/Downloads/dast_poc/output"
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "Parser_html.dast.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(root.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"DAST saved to {output_file}")
        return root
    
if __name__ == "__main__":
    html_file = "/Users/anushkaupadhyay/Downloads/dast_poc/demo/sample.html"

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    parser = HTMLParser()
    root = parser.parse(
        html_content,
        doc_title=os.path.basename(html_file)
    )

    print(f"Parsed document: {root.title}")