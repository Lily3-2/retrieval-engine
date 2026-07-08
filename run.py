# #Step 1: parse document2 -> DAST
# from parsers.pdf_parser import PDFParser
# import json

# r=PDFParser().parse('benchmark/document2.pdf')
# json.dump(r.to_dict(), open('benchmark/document2.dast.json', 'w'), indent=2)
# print('wrote benchmark/document2.dast.json')


# #Step2: browse the tree to pick real node_ids for our questions
# # this is the helper we will use to author ground-truth. Search the parsed doc for text, and it prints the mtching node_ids (which we paste into questions.jsonl)
# d=json.load(open('benchmark/document2.dast.json'))
# def walk(n):
#   yield n
#   for c in n['children']: yield from walk(c)
# term=input('search term: ').lower()
# for n in walk(d):
#   txt=((n.get('title') or '')+' '+(n.get('text') or '')).lower()
#   if term in txt:
#     loc=n.get('physical_location') or {}
#     print(n['node_id'], '| p',loc.get('page_start'),'|',(n.get('title') or n.get('text') or '')[:80].replace(chr(10),' '))

# #eg: type: job title  -> give a doc.10.11.6 (p7) etc. to use as ground_truth_node_ids


# # Step3: author benchmark/questions.jsonl
# # cat > benchmark/questions.jsonl <<'EOF'
# # {"qid":"q001","question":"What should a good job title look like?","answer":"Clear, plain, and searchable — drop the internal jargon and formality.","ground_truth_node_ids":["doc.19.11.6"],"answer_type":"factual","requires_table":false,"requires_xref":false}
# # {"qid":"q002","question":"How should you describe what the role involves?","answer":"Describe the day-to-day of the job the way you'd explain it to a friend.","ground_truth_node_ids":["doc.19.12.2.3"],"answer_type":"factual","requires_table":false,"requires_xref":false}
# # EOF

# # ^ these two are STARTERS with real node_ids from your doc. Add 15-30 total, verifying each grond_truth_node_id against Step 2 output.


#Step3.1: engine-only sanity check (no LLM, instant)
from engine.index import DASTIndex
from engine.retriever import DASTRetriever
idx=DASTIndex.from_json('benchmark/document2.dast.json')
for x in DASTRetriever(idx).retrieve('tips for writing job descriptions', top_k=3):
  print(round(x.score,3), x.node_id, x.matched_terms)


# Step 4: the full bake-off (DAST vs RAG baseline)
export HF_TOKEN=open('D:/Mine/Post-grad-prep/fine_grain_hf_token.txt') # your HuggingFace token (Llama-3.2 needs it)

python3 -m eval.run_benchmark \
--dast benchmark/document2.dast.json\
--questions benchmark/questions.jsonl\
--pdf benchmark/document2.pdf \
--output-dir results \
--top-k 8

# results land in results/run_<timestamp>.json -> open the "summary" block:  
#                                                                             
import json,glob
f=sorted(glob.glob('results/run_*.json'))[-1];
print(json.dumps(json.load(open(f))['summary'], indent=2))"