from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkExample:
    qid: str
    question: str
    answer: str
    ground_truth_node_ids: List[str]
    answer_type: str
    requires_table: bool
    requires_xref: bool


def load_dataset(path: str) -> List[BenchmarkExample]:
    examples: List[BenchmarkExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.append(BenchmarkExample(
                qid=payload["qid"],
                question=payload["question"],
                answer=payload.get("answer", ""),
                ground_truth_node_ids=payload.get("ground_truth_node_ids", []),
                answer_type=payload.get("answer_type", ""),
                requires_table=payload.get("requires_table", False),
                requires_xref=payload.get("requires_xref", False),
            ))
    return examples


def validate_dataset(examples: List[BenchmarkExample]) -> None:
    for example in examples:
        if not example.qid:
            raise ValueError("Each benchmark example must have a qid")
        if not example.question:
            raise ValueError(f"Example {example.qid} is missing a question")
        if example.ground_truth_node_ids is None:
            raise ValueError(f"Example {example.qid} must include ground_truth_node_ids")
