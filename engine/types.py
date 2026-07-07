from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ScoredNode:
    node_id: str
    score: float
    factors: Dict[str, float]
    matched_terms: List[str]
    semantic_path: List[str]
    physical_location: Dict[str, Any]
    text: str
