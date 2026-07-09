# #Step 1: parse document2 -> DAST
from parsers.pdf_parser import PDFParser
import json

r=PDFParser().parse('benchmark/document2.pdf')
json.dump(r.to_dict(), open('benchmark/document2.dast.json', 'w'), indent=2)
print('wrote benchmark/document2.dast.json')


# #Step2: browse the tree to pick real node_ids for our questions
# # this is the helper we will use to author ground-truth. Search the parsed doc for text, and it prints the mtching node_ids (which we paste into questions.jsonl)
d=json.load(open('benchmark/document2.dast.json'))
def walk(n):
  yield n
  for c in n['children']: yield from walk(c)
term=input('search term: ').lower()
for n in walk(d):
  txt=((n.get('title') or '')+' '+(n.get('text') or '')).lower()
  if term in txt:
    loc=n.get('physical_location') or {}
    print(n['node_id'], '| p',loc.get('page_start'),'|',(n.get('title') or n.get('text') or '')[:80].replace(chr(10),' '))

# #eg: type: job title  -> give a doc.10.11.6 (p7) etc. to use as ground_truth_node_ids



# # ^ these two are STARTERS with real node_ids from your doc. Add 15-30 total, verifying each grond_truth_node_id against Step 2 output.


#Step3.1: engine-only sanity check (no LLM, instant)
from engine.index import DASTIndex
from engine.retriever import DASTRetriever

def related (a, b):
  return a == b or b.startswith(a + ".") or a.startswith(b + ".")


idx=DASTIndex.from_json('benchmark/document2.dast.json')
eng = DASTRetriever(idx, startegy="best_first")

strict = lenient = total = 0
for line in open('benchmark/questions.jsonl'):
  line = line.strip()
  if not line: continue
  ex = json.loads(line)
  total += 1
  got = [r.node_id for r in eng.retrieve(ex['question'], top_k=8)]
  gt = ex['ground_truth_node_ids'][0]
  s = gt in got
  l = any(related(n, gt) for n in got)
  strict += s
  lenient += l
  print(("L-HIT" if l else "MISS "), "S" if s else ".", ex["qid"], "| gt", gt, "| top5", got[:5])

print(f"\nStrict hits: {strict}/{total}  |  ANCESTOR/DESCENDANT hit@8: {lenient}/{total}")
for x in DASTRetriever(idx).retrieve('tips for writing job descriptions', top_k=3):
  print(round(x.score,3), x.node_id, x.matched_terms)


# Step 4: the full bake-off (DAST vs RAG baseline) (needs deps + token)
# install tesseract
import os
from eval.run_benchmark import run_benchmark

result = run_benchmark(
  dast_path='benchmark/document2.dast.json',
  questions_path='benchmark/questions.jsonl',
  pdf_path='benchmark/document2.pdf',
  output_dir='results',
  top_k=8,
)
print("Benhmark complete:", result)



# pretty-print the summary block
data = json.load(open(result['output_dir']))
print("\n=== SUMMARY ===")
print(json.dumps(data['summary'], indent=2))