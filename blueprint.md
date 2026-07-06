- determisnistic parsing quality is bounded by the layout heuristic 


Blueprint: 

Guiding Constraint 
The whole thesis: no LLM in the retrieval loop. The LLM is called explicitly one, at the very end, to phrase the answer from evidence the deterministic engine already picked. If you ever find yourself asking the LLM "is this node relevant?", you've become PageIndex and lost the experiment.

proposed file layout:
retrieval-engine/
  dast_schema.py            # exists
  xref_resolver.py          # exists
  parsers/                  # exists
  RAG/                      # exists (FAISS baseline)

  # --- NEW: Task 2 ---
  engine/
    init.py
    index.py                # in-memory index built from a DAST tree
    scoring.py              # pure scoring functions (the research core)
    traversal.py            # BFS / DFS / best-first strategies
    retriever.py            # orchestrator: query -> ranked nodes + explanations
    entities.py             # deterministic entity/keyword extraction (no ML)

  # --- NEW: Task 3 ---
  eval/
    context_builder.py      # ranked nodes -> assembled context string
    prompts.py              # answer-generation prompt templates
    generator.py            # thin LLM wrapper (the ONLY LLM call)
    dataset.py              # load/validate benchmark dataset
    metrics.py              # retrieval + answer + latency + token metrics
    run_benchmark.py        # main entrypoint: DAST vs FAISS baseline
    baseline_adapter.py     # wraps RAG/ IndexManager to a common interface

  benchmark/
    document.pdf            # the ONE shared doc (Attention paper)
    document.dast.json      # parsed once, cached
    questions.jsonl         # ground-truth Q&A dataset
  results/
    run_<timestamp>.json    # raw metrics per run
    report.html             # optional flat report


2. Data schemas (the contracts everything shares)
─────────────────────────────────────────────────

2.1 Query object
 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass                                                                                                                                                                                     
class Query:                                                                                                                                                                                   
    text: str                                                                                                                                                                                  
    keywords: list[str]        # extracted, lowercased, stemmed                                                                                                                                
    entities: list[str]        # capitalized terms, numbers+units, quoted spans                                                                                                                
    raw_refs: list[str]        # detected "Section 3", "Table 1" in the query itself                                                                                                           
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

2.2 Scored result (what retrieval returns — must be fully explainable)
 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass                                                                                                                                                                                     
class ScoredNode:                                                                                                                                                                              
    node_id: str                                                                                                                                                                               
    score: float                       # final combined score                                                                                                                                  
    factors: dict[str, float]          # per-factor breakdown, e.g.                                                                                                                            
                                       # {"keyword":.4,"structural":.1,"entity":.2,                                                                                                            
                                       #  "xref":.05,"ancestor":.1,"path":.15}                                                                                                                 
    matched_terms: list[str]           # which query terms hit this node                                                                                                                       
    semantic_path: list[str]           # copied from node for readability                                                                                                                      
    physical_location: dict            # copied for citation/tracing                                                                                                                           
    text: str                          # node text (for context builder)                                                                                                                       
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
The factors dict is non-negotiable — it's your explainability metric and the thing that lets Task 3 answer "can every answer be traced to explicit AST nodes?"

2.3 Benchmark dataset row (questions.jsonl, one JSON per line)
 json ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
{                                                                                                                                                                                              
  "qid": "q001",                                                                                                                                                                               
  "question": "What is the dimensionality of the keys used in scaled dot-product attention?",                                                                                                  
  "answer": "64",                                                                                                                                                                              
  "ground_truth_node_ids": ["doc.3.2.1", "doc.3.2.1.4"],                                                                                                                                       
  "answer_type": "factual",                                                                                                                                                                    
  "requires_table": false,                                                                                                                                                                     
  "requires_xref": false                                                                                                                                                                       
}                                                                                                                                                                                              
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
ground_truth_node_ids is what makes retrieval measurable. Author 15–30 of these by hand against the parsed tree (open document.dast.json, find the nodes that truly contain each answer, copy their node_ids).



─────────────────────────────────────────────────

3. engine/index.py — the in-memory index
────────────────────────────────────────

Purpose: load a DAST JSON once, build lookups so scoring/traversal is O(1) per node.

 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class DASTIndex:                                                                                                                                                                               
    def _init_(self, root: DASTNode): ...                                                                                                                                                    
                                                                                                                                                                                               
    @classmethod                                                                                                                                                                               
    def from_json(cls, path: str) -> "DASTIndex": ...                                                                                                                                          
                                                                                                                                                                                               
    # lookups                                                                                                                                                                                  
    nodes_by_id: dict[str, DASTNode]                                                                                                                                                           
    parent_of: dict[str, str | None]                                                                                                                                                           
    children_of: dict[str, list[str]]                                                                                                                                                          
    xref_edges: dict[str, list[str]]      # node_id -> resolved target ids                                                                                                                     
                                                                                                                                                                                               
    # precomputed corpus stats for scoring                                                                                                                                                     
    term_doc_freq: dict[str, int]         # for IDF-style weighting                                                                                                                            
    total_nodes: int                                                                                                                                                                           
                                                                                                                                                                                               
    def ancestors(self, node_id: str) -> list[str]: ...   # via node_id string-trim                                                                                                            
    def descendants(self, node_id: str) -> list[str]: ...                                                                                                                                      
    def depth(self, node_id: str) -> int: ...             # node_id.count(".")                                                                                                                 
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Key trick: because node_id encodes the path (doc.2.1.3), parent_of and ancestors are pure string operations — no tree walking needed. Lean on that.


─────────────────────────────────────────────────

4. engine/entities.py — deterministic extraction (NO ML)
────────────────────────────────────────────────────────

 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]: ...          # lowercase, split, strip punct                                                                                                             
def stem(tokens: list[str]) -> list[str]: ...       # Porter stemmer (nltk) or simple suffix rules                                                                                             
def extract_keywords(text: str, stopwords: set[str]) -> list[str]: ...                                                                                                                         
def extract_entities(text: str) -> list[str]:                                                                                                                                                  
    # rule-based: Capitalized multi-word spans, numbers+units (64, 512-dim,                                                                                                                    
    # 8 heads), quoted phrases, acronyms (BLEU, GELU). Regex only.                                                                                                                             
def build_query(text: str) -> Query: ...            # combines the above + xref detect                                                                                                         
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Everything deterministic and inspectable. This is what keeps you honest vs the embedding baseline.

───────────────────────────────────────────────────────

5. engine/scoring.py — THE research core
────────────────────────────────────────

Six pure functions, each (query, node, index) -> float in [0,1]. Keep them pure so you can unit-test and ablate each independently (that directly answers "which structural signals contribute most?").

 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def keyword_score(q: Query, node, index) -> float:                                                                                                                                             
    # TF of query keywords in node.text, weighted by IDF from index.term_doc_freq,                                                                                                             
    # normalized by node length. Classic BM25-lite is ideal here.                                                                                                                              
                                                                                                                                                                                               
def structural_score(q: Query, node, index) -> float:                                                                                                                                          
    # weight by node_type: SECTION title > PARAGRAPH > LIST_ITEM > TABLE_CELL.                                                                                                                 
    # Use a fixed NODE_TYPE_WEIGHT dict. Also boost shallow depth slightly.                                                                                                                    
                                                                                                                                                                                               
def entity_score(q: Query, node, index) -> float:                                                                                                                                              
    # overlap between q.entities and entities extracted from node (exact + numeric).                                                                                                           
                                                                                                                                                                                               
def xref_score(q: Query, node, index) -> float:                                                                                                                                                
    # if node is the resolved target of an xref, OR the query names a ref that                                                                                                                 
    # resolves to this node, boost. Follows index.xref_edges.                                                                                                                                  
                                                                                                                                                                                               
def ancestor_descendant_score(q: Query, node, index) -> float:                                                                                                                                 
    # keyword hits in ancestors (section titles) and immediate children.                                                                                                                       
    # "answer under 3.2 Attention" -> section title match propagates down.                                                                                                                     
                                                                                                                                                                                               
def path_score(q: Query, node, index) -> float:                                                                                                                                                
    # keyword overlap with node.semantic_path breadcrumb (title chain).                                                                                                                        
───────────────────────────────────────────


Combination formula (start linear, tune weights)
 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {                                                                                                                                                                            
    "keyword": 0.35, "structural": 0.10, "entity": 0.20,                                                                                                                                       
    "xref": 0.05, "ancestor": 0.15, "path": 0.15,                                                                                                                                              
}                                                                                                                                                                                              
final = sum(w[k] * factors[k] for k in factors)   # then min-max normalize across candidates                                                                                                   
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Design note for your writeup — this directly answers "how should multiple factors be combined?": start with a fixed linear weighted sum (fully explainable). Only if that underperforms, try (a) learned weights via grid-search on the benchmark, or (b) a lexicographic tie-break (structural first, keyword as tiebreaker). Document which won.

──────────────────

6. engine/traversal.py — strategies (answers "BFS vs DFS vs best-first")
────────────────────────────────────────────────────────────────────────

 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def bfs(index, q, score_fn, top_k) -> list[ScoredNode]: ...                                                                                                                                    
def dfs(index, q, score_fn, top_k, prune_threshold=0.0) -> list[ScoredNode]: ...                                                                                                               
def best_first(index, q, score_fn, top_k, beam=None) -> list[ScoredNode]:                                                                                                                      
    # priority queue ordered by node score; expand children of high-scoring                                                                                                                    
    # sections first, prune subtrees whose section score < threshold.                                                                                                                          
──────────────────
Best-first is where your thesis shines: score a SECTION, and if it's cold, prune its entire subtree without scoring descendants — deterministic, fast, explainable. Benchmark all three on latency + accuracy; report the tradeoff.

7. 7. engine/retriever.py — orchestrator
─────────────────────────────────────

 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class DASTRetriever:                                                                                                                                                                           
    def _init_(self, index: DASTIndex, weights=DEFAULT_WEIGHTS,                                                                                                                              
                 strategy="best_first"): ...                                                                                                                                                   
                                                                                                                                                                                               
    def retrieve(self, question: str, top_k: int = 8) -> list[ScoredNode]:                                                                                                                     
        # 1. build_query(question)                                                                                                                                                             
        # 2. run chosen traversal with combined score_fn                                                                                                                                       
        # 3. return top_k ScoredNode WITH factor breakdowns                                                                                                                                    
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
This is the drop-in counterpart to RAG/IndexManager.query() — same shape (top_k results), different soul.

─────────────────────────────────────────────────────────

8. eval/baseline_adapter.py — make the comparison fair
──────────────────────────────────────────────────────

Wrap the existing FAISS baseline behind the same interface so run_benchmark.py treats both identically:

 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class BaselineRetriever:                                                                                                                                                                       
    def _init_(self, pdf_path: str): ...      # runs RAG PDFProcessor + IndexManager                                                                                                         
    def retrieve(self, question: str, top_k: int = 8) -> list[dict]:                                                                                                                           
        # returns [{"text":..., "info":"file.pdf 3", "score":dist}, ...]                                                                                                                       
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Reconciliation gotcha: the baseline returns (chunk_text, "file page") — it has no node_ids. So for retrieval-accuracy scoring you must map baseline chunks → ground-truth via page-number overlap or text containment (does the chunk contain the ground-truth node's text?), since it can't produce node_ids. Document this asymmetry honestly — it's itself a finding (the baseline literally cannot cite structure).

───────────────────────

9. eval/context_builder.py (answers "how much surrounding context?")
────────────────────────────────────────────────────────────────────

 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def build_context(results: list[ScoredNode], index: DASTIndex,                                                                                                                                 
                  policy: str = "node_plus_parent",                                                                                                                                            
                  max_tokens: int = 2000) -> tuple[str, list[str]]:                                                                                                                            
    # policies to A/B:                                                                                                                                                                         
    #   "node_only"          -> just the hit text                                                                                                                                              
    #   "node_plus_parent"   -> hit + its section title/intro                                                                                                                                  
    #   "node_plus_siblings" -> hit + adjacent siblings (for lists/tables)                                                                                                                     
    #   "subtree"            -> whole section subtree of the hit                                                                                                                               
    # returns (context_string, list_of_included_node_ids_for_tracing)                                                                                                                          
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Run the benchmark across policies to quantitatively answer the "how much context" research question. Return the included node_ids so every answer is traceable (Task 3 requirement).

───────────────

10. eval/prompts.py + generator.py (the ONE LLM call)
─────────────────────────────────────────────────────

 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
ANSWER_PROMPT = """You are answering strictly from the provided structured evidence.                                                                                                           
Each evidence block is labeled with its node_id and semantic path.                                                                                                                             
Answer concisely. After your answer, cite the node_id(s) you used.                                                                                                                             
If the evidence does not contain the answer, say "Not found in document."                                                                                                                      
                                                                                                                                                                                               
Evidence:                                                                                                                                                                                      
{context}                                                                                                                                                                                      
                                                                                                                                                                                               
Question: {question}                                                                                                                                                                           
Answer:"""                                                                                                                                                                                     
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def generate_answer(question, context) -> dict:                                                                                                                                                
    # returns {"answer":..., "cited_node_ids":[...], "prompt_tokens":n, "completion_tokens":m}                                                                                                 
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Keep it a thin wrapper (reuse RAG/LargeLanguageModel or any API). Log token counts — that's a required Task 3 metric.


11. eval/metrics.py
───────────────────

 python ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def retrieval_metrics(retrieved_ids, ground_truth_ids, k) -> dict:                                                                                                                             
    # precision@k, recall@k, MRR, hit@k                                                                                                                                                        
def answer_metrics(pred, gold) -> dict:                                                                                                                                                        
    # exact_match, F1 (token overlap), plus optional LLM-judge faithfulness                                                                                                                    
def traceability(cited_ids, ground_truth_ids) -> float:                                                                                                                                        
    # fraction of answers whose citations include a true evidence node                                                                                                                         
def efficiency(latency_s, prompt_tokens, completion_tokens) -> dict:                                                                                                                           
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Metrics to report (straight from the problem statement): retrieval accuracy, answer accuracy, latency, explainability, token usage. Explainability = you have the factor breakdown + cited node_ids; the baseline has neither. That's your headline result.

──────────────────────────────

12. eval/run_benchmark.py — the experiment
──────────────────────────────────────────

for each question in dataset:
    for engine in [DASTRetriever, BaselineRetriever]:
        results = engine.retrieve(q, top_k)
        log retrieval_metrics vs ground_truth
        context, used_ids = build_context(results)      # DAST only for structured policies
        ans = generate_answer(q, context)
        log answer_metrics, traceability, latency, tokens
aggregate -> results/run_<ts>.json -> optional report.html

Then run ablations (the part that makes it research, not a demo):
* turn each scoring factor off, one at a time → which matters most?
* BFS vs DFS vs best-first → latency/accuracy tradeoff
* context policy sweep → how much surrounding context is optimal?

────────────────

13. Recommended build order
───────────────────────────

* benchmark/: parse the Attention PDF → document.dast.json, eyeball the tree (check the 2-column heading damage), author questions.jsonl (15–30 Qs) with real node_ids.
* engine/: index.py → entities.py → scoring.py (one factor at a time, unit-test each) → traversal.py → retriever.py.
* eval/baseline_adapter.py: get FAISS running on the same PDF, solve the chunk→node_id mapping.
* eval/: metrics.py → context_builder.py → prompts.py/generator.py → run_benchmark.py.
* Ablations + writeup: this is what proves the final verdict.

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

14. The verdict you're trying to prove
──────────────────────────────────────
│ Can we eliminate LLM-driven retrieval by encoding semantics into a deterministic DAST + rule engine?

You'll answer it with three numbers: (a) DAST retrieval precision@k ≥ FAISS baseline, (b) answer accuracy within ~epsilon of the baseline, (c) DAST offers 100% traceability + factor-level explainability while the baseline offers ~0%. If (a) and (b) hold and (c) is a landslide → thesis proven.