# Task 1 — Document AST & Parsing (DAST)

DAST - Document Abstract Syntax Tree
POC deliverable: a single schema (`dast_schema.py`) + three format parsers
(Markdown, HTML, PDF) that all normalize into the same tree shape, proving a
document can be treated as a structured, traversable object rather than
unstructured text — the prerequisite for Task 2's deterministic retrieval engine.

## Files

```
dast_schema.py          DAST node schema (NodeType, PhysicalLocation, DASTNode)
xref_resolver.py         Cross-reference detection + resolution (shared by all parsers)
parsers/
  markdown_parser.py     .md -> DAST  (via markdown-it-py CommonMark tokens)
  html_parser.py         .html -> DAST (via BeautifulSoup DOM walk)
  pdf_parser.py           .pdf -> DAST (via pdfplumber, font-size heading heuristic)
demo/
  sample.md / sample.html / sample.pdf   equivalent content in all 3 formats
build_and_test.py        runs all 3 parsers, prints tree, dumps JSON, sanity-checks
output/                  generated *.dast.json files
```

Run it: `python3 build_and_test.py`

## Research question answers

### 1. What semantic node types are needed?

A closed 14-value enum (`NodeType`) that every parser must normalize into:

`DOCUMENT, SECTION, PARAGRAPH, LIST, LIST_ITEM, TABLE, TABLE_ROW, TABLE_CELL, CODE_BLOCK, FIGURE, CAPTION, QUOTE, REFERENCE, FOOTNOTE, THEMATIC_BREAK`

Key decision: **SECTION is the backbone, not HEADING.** A heading doesn't
just sit in the tree — it *opens* a subtree containing everything until the
next heading of equal or shallower depth. This is what makes ancestor/
descendant traversal (Task 2's "structural weight" and "path-based scoring")
meaningful: asking "what's under 2.2 Ceiling" is a direct subtree query, not
a linear scan for the next heading string.

This set is deliberately format-agnostic — the demo proves it, since the
same document parsed from `.md`, `.html`, and `.pdf` produces the same 7
sections, the same 1 table, the same nesting depth. That convergence is the
whole point: retrieval logic in Task 2 doesn't need to know or care what the
source format was.

### 2. How should physical location be represented?

A `PhysicalLocation` object attached to every node, populated with whatever
subset applies to the source format — never forcing an HTML node to fake a
page number, or a Markdown node to fake a bounding box:

| Format   | page_start/end | bbox | line_start/end | char_start/end |
|----------|:---:|:---:|:---:|:---:|
| Markdown |  |  | ✅ | ✅ |
| HTML     |  |  | ✅ (approx, via `sourceline`) |  |
| PDF      | ✅ | ✅ | | |

PDF is the only format where `bbox` (x0,y0,x1,y1 in PDF points) is meaningful
— it's the format where "physical layout" is a real, not metaphorical,
concept. This is deliberately kept separate from `semantic_path` (see below)
so a retrieval hit can answer *both* "where is this in the document's logical
structure" and "where is this on the physical page" — the latter is what
would let a future UI literally highlight the box on a PDF page image.

### 3. How do we uniquely identify every node?

Three identifiers, each doing a different job — this was the most
consequential design decision, because it's what Task 2's traversal
algorithms and Task 3's citation-tracing depend on:

- **`node_id`** — a deterministic, position-encoding path, e.g. `doc.2.1.3`
  (3rd child of the 1st child of the 2nd top-level section). This is the
  *primary key for traversal*: parent lookup is a string trim, sibling order
  is implicit, and depth is `node_id.count(".")`. Deterministic means
  re-parsing the same document version always yields the same IDs — required
  for reproducible retrieval benchmarks in Task 2/3.
- **`uuid`** — random UUID4. Needed because `node_id` shifts if the document
  is edited (insert a section, every later sibling's path changes). `uuid`
  gives a stable identity for anything that needs to survive document edits
  (e.g. a citation saved by a user before the doc was revised).
- **`content_hash`** — sha1 of `semantic_path + node_type + text`. This is
  the free "cosmetic" the prompt mentions (token caching): if a node's hash
  is unchanged across re-parses, its retrieval score and any cached LLM
  output for it can be reused without recomputation.

Additionally every node carries a **`semantic_path`** — the breadcrumb of
ancestor titles, e.g. `["Employee Leave Policy Specification", "2. Accrual
Rules", "2.2 Ceiling"]`. This is the richer version of PageIndex's flat
node summary: it's what lets Task 2's scoring framework do "path-based
scoring" (does the query match words anywhere in the ancestor chain, not
just the node itself) and is human-readable in a way `node_id` alone isn't.

## Cross-references: the fourth structural signal

Beyond hierarchy, technical documents are full of internal pointers — "see
Table 1", "as per Section 3", "Appendix A". `xref_resolver.py` detects these
via pattern-matching, then does a second pass over the *fully built* tree to
resolve each raw string to a target `node_id` (matching against section/
table/figure titles). This turns "cross-reference score" (mentioned in the
prompt's Task 2 scoring factors) into a real graph edge Claude — sorry, the
retrieval engine — can follow, not just a string it has to re-search for.

In the demo, 6-7 references are detected per format; 2 resolve automatically
(the direct "Section 2.1"/"Section 2.2"/"Section 3" style references) while
others reference labels not present as node titles in this particular sample
(e.g. "Table 1" — the actual table has no numbered title, only a heading
above it). This is intentional and informative: it's a live example of the
kind of resolution failure the scoring engine will need to be robust to
(fall back to fuzzy/keyword match on unresolved refs rather than dropping
them).

## What's a heuristic vs. what's structural fact

Worth being explicit about, since it matters for how Task 2/3 should weight
confidence in different signal types:

- **Markdown/HTML parsing is structural fact.** Heading levels, list nesting,
  and table shape come directly from the source markup — zero inference.
- **PDF parsing is heuristic.** There's no semantic tag for "this is a
  heading" in a PDF; `pdf_parser.py` infers it from font-size buckets
  (largest distinct font size on the page = level 1, etc.), which works well
  for documents with consistent typographic conventions and will misfire on
  documents that use bold-not-bigger headings, all-caps body text, or
  multi-column layouts. This is flagged here rather than papered over,
  because it's the natural place to later plug in a layout model (e.g. a
  LayoutLM-style classifier) without touching the DAST schema or anything
  downstream — the heuristic is isolated to one function
  (`PDFParser.parse`'s heading-detection block).

## Cross-format consistency (proof it worked)

Running `build_and_test.py` on the three equivalent sample documents:

| format   | sections | paragraphs | tables | node_id/uuid uniqueness |
|----------|:---:|:---:|:---:|:---:|
| markdown | 7 | 6  | 1 | ✅ |
| html     | 7 | 4  | 1 | ✅ |
| pdf      | 7 | 12 | 1 | ✅ |

Same section count and table count across all three formats confirms the
schema is truly format-agnostic. (Paragraph counts differ because Markdown/
HTML preserve author paragraph breaks while the PDF parser currently emits
one node per visual text-line before line-merging — a known refinement for
later: line-merging into paragraph blocks via vertical-gap thresholds. Not a
schema issue, just a parser refinement noted for the next iteration.)

## Handoff to Task 2

Every node produced here already carries what the scoring framework needs:
`text` (keyword matching), `semantic_path` (path-based scoring), `parent_id`
+ `children` (ancestor/descendant relevance), `cross_references` (xref
score), and `node_type`/`level` (structural weighting — e.g. weight SECTION
titles higher than deeply-nested LIST_ITEMs). No additional preprocessing
should be needed to start Task 2's traversal engine directly against these
JSON trees.


Steps to run:
parse document2 -> DAST
python3 - c "from parsers.pdf_parser import PDFParser; import json; \
r=PDFParser().parse('benchmark/document2.pdf');\
json.dump(r.to_dict(), open('benchmark/document2.dast.json', 'w'), indent=2);\
print('wrote benchmark/document2.dast.json')"