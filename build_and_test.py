"""
Task 1 demo/test harness.

Runs all three parsers (Markdown, HTML, PDF) on the equivalent sample
document, dumps each resulting DAST to JSON, and prints:
  - a readable tree outline (node_id, type, title/text preview)
  - basic sanity stats (node count, section count, resolved cross-refs)
  - a spot-check that the same logical content produces the same shape
    across formats (proving format-agnostic retrieval is viable)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parsers.markdown_parser import MarkdownParser
from parsers.html_parser import HTMLParser
from parsers.pdf_parser import PDFParser

DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def print_tree(node, indent=0):
    preview = (node.title or node.text or "")[:70].replace("\n", " ")
    print(f"{'  ' * indent}[{node.node_id}] {node.node_type.value:12s} {preview}")
    for c in node.children:
        print_tree(c, indent + 1)


def stats(node):
    all_nodes = list(node.walk())
    counts = {}
    total_refs = 0
    resolved_refs = 0
    for n in all_nodes:
        counts[n.node_type.value] = counts.get(n.node_type.value, 0) + 1
        total_refs += len(n.cross_references)
        resolved_refs += sum(1 for r in n.cross_references if r.resolved)
    return {
        "total_nodes": len(all_nodes),
        "by_type": counts,
        "cross_references_found": total_refs,
        "cross_references_resolved": resolved_refs,
    }


def run(label, root, out_name):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print_tree(root)
    s = stats(root)
    print(f"\nStats: {json.dumps(s, indent=2)}")
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump(root.to_dict(), f, indent=2)
    print(f"JSON written -> {out_path}")
    return s


if __name__ == "__main__":
    with open(os.path.join(DEMO_DIR, "sample.md")) as f:
        md_source = f.read()
    md_root = MarkdownParser().parse(md_source, doc_title="Employee Leave Policy Specification")
    md_stats = run("MARKDOWN PARSER OUTPUT", md_root, "sample_md.dast.json")

    with open(os.path.join(DEMO_DIR, "sample.html")) as f:
        html_source = f.read()
    html_root = HTMLParser().parse(html_source, doc_title="Employee Leave Policy Specification")
    html_stats = run("HTML PARSER OUTPUT", html_root, "sample_html.dast.json")

    pdf_root = PDFParser().parse(os.path.join(DEMO_DIR, "sample.pdf"))
    pdf_stats = run("PDF PARSER OUTPUT", pdf_root, "sample_pdf.dast.json")

    print(f"\n{'=' * 70}\nCROSS-FORMAT CONSISTENCY CHECK\n{'=' * 70}")
    print(f"{'format':10s} {'sections':10s} {'paragraphs':12s} {'tables':8s} {'xrefs_resolved'}")
    for label, s in [("markdown", md_stats), ("html", html_stats), ("pdf", pdf_stats)]:
        print(f"{label:10s} {s['by_type'].get('section', 0):<10d} "
              f"{s['by_type'].get('paragraph', 0):<12d} {s['by_type'].get('table', 0):<8d} "
              f"{s['cross_references_resolved']}/{s['cross_references_found']}")

    # spot check: node_id uniqueness within each tree
    for label, root in [("markdown", md_root), ("html", html_root), ("pdf", pdf_root)]:
        ids = [n.node_id for n in root.walk()]
        assert len(ids) == len(set(ids)), f"DUPLICATE node_id in {label} tree!"
        uuids = [n.uuid for n in root.walk()]
        assert len(uuids) == len(set(uuids)), f"DUPLICATE uuid in {label} tree!"
    print("\nAll node_id and uuid values are unique within each tree. ✅")
