from __future__ import annotations

from typing import Dict, List


def retrieval_metrics(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> Dict[str, float]:
    retrieved_set = set(retrieved_ids[:k])
    ground_truth_set = set(ground_truth_ids)
    true_positives = len(retrieved_set & ground_truth_set)
    precision = true_positives / max(1, k)
    recall = true_positives / max(1, len(ground_truth_set))
    rr = 0.0
    for rank, node_id in enumerate(retrieved_ids[:k], start=1):
        if node_id in ground_truth_set:
            rr = 1.0 / rank
            break
    hit = 1.0 if true_positives > 0 else 0.0
    return {
        "precision@k": precision,
        "recall@k": recall,
        "mrr": rr,
        "hit@k": hit,
    }


def answer_metrics(prediction: str, gold: str) -> Dict[str, float]:
    lower_pred = prediction.strip().lower()
    lower_gold = gold.strip().lower()
    exact_match = 1.0 if lower_pred == lower_gold else 0.0
    pred_tokens = lower_pred.split()
    gold_tokens = lower_gold.split()
    common = set(pred_tokens) & set(gold_tokens)
    f1 = (2 * len(common) / (len(pred_tokens) + len(gold_tokens))) if pred_tokens and gold_tokens else 0.0
    return {
        "exact_match": exact_match,
        "f1": f1,
    }


def traceability(cited_ids: List[str], ground_truth_ids: List[str]) -> float:
    if not cited_ids or not ground_truth_ids:
        return 0.0
    cited_set = set(cited_ids)
    truth_set = set(ground_truth_ids)
    return len(cited_set & truth_set) / len(truth_set)


def efficiency(latency_s: float, prompt_tokens: int, completion_tokens: int) -> Dict[str, float]:
    return {
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
