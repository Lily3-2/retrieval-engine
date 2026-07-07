from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from engine.index import DASTIndex
from engine.retriever import DASTRetriever
from .context_builder import build_context
from .dataset import load_dataset, validate_dataset
from .generator import AnswerGenerator
from .metrics import answer_metrics, baseline_retrieval_metrics, efficiency, retrieval_metrics, traceability
from .baseline_adapter import BaselineRetriever


def run_benchmark(dast_path: str, questions_path: str, pdf_path: str, output_dir: str, top_k: int = 8) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    index = DASTIndex.from_json(dast_path)
    retriever = DASTRetriever(index)
    generator = AnswerGenerator()
    baseline = BaselineRetriever(pdf_path)
    examples = load_dataset(questions_path)
    validate_dataset(examples)

    run_results: List[Dict[str, Any]] = []

    for example in examples:
        start = time.time()
        results = retriever.retrieve(example.question, top_k=top_k)
        duration = time.time() - start
        retrieved_ids = [item.node_id for item in results]
        retrieval_stats = retrieval_metrics(retrieved_ids, example.ground_truth_node_ids, top_k)
        context, cited_ids = build_context(results, index, policy="node_plus_parent")
        answer_data = generator.generate_answer(example.question, context)
        answer_stats = answer_metrics(answer_data["answer"], example.answer)
        trace_score = traceability(answer_data["cited_node_ids"], example.ground_truth_node_ids)
        baseline_results = baseline.retrieve(example.question, top_k=top_k)
        baseline_stats = baseline_retrieval_metrics(baseline_results, example.ground_truth_node_ids, index, top_k)
        run_results.append({
            "qid": example.qid,
            "question": example.question,
            "ground_truth": example.ground_truth_node_ids,
            "retrieved_ids": retrieved_ids,
            "retrieval_metrics": retrieval_stats,
            "answer": answer_data["answer"],
            "answer_metrics": answer_stats,
            "traceability": trace_score,
            "efficiency": efficiency(duration, answer_data["prompt_tokens"], answer_data["completion_tokens"]),
            "baseline_results": baseline_results,
            "baseline_metrics": baseline_stats,
        })

    def _mean(rows: List[Dict[str, Any]], path: tuple[str, str]) -> float:
        vals = [r[path[0]][path[1]] for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "dast_precision@k": _mean(run_results, ("retrieval_metrics", "precision@k")),
        "dast_recall@k": _mean(run_results, ("retrieval_metrics", "recall@k")),
        "baseline_precision@k": _mean(run_results, ("baseline_metrics", "precision@k")),
        "answer_f1": _mean(run_results, ("answer_metrics", "f1")),
        "mean_traceability": sum(r["traceability"] for r in run_results) / len(run_results) if run_results else 0.0,
        "mean_latency_s": _mean(run_results, ("efficiency", "latency_s")),
    }

    output_path = os.path.join(output_dir, f"run_{int(time.time())}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": run_results}, f, indent=2)
    return {"output_path": output_path, "num_examples": len(run_results)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DAST vs baseline benchmark.")
    parser.add_argument("--dast", required=True, help="Path to document.dast.json")
    parser.add_argument("--questions", required=True, help="Path to questions.jsonl")
    parser.add_argument("--pdf", required=True, help="Path to benchmark PDF or containing folder")
    parser.add_argument("--output-dir", default="results", help="Directory for raw benchmark output")
    parser.add_argument("--top-k", type=int, default=8, help="Number of results to retrieve")
    args = parser.parse_args()

    result = run_benchmark(args.dast, args.questions, args.pdf, args.output_dir, top_k=args.top_k)
    print(f"Benchmark complete: {result}")
