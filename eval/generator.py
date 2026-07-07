from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .prompts import ANSWER_PROMPT


class AnswerGenerator:
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def generate_answer(self, question: str, context: str, max_new_tokens: int = 200, temperature: float = 0.0) -> Dict[str, Any]:
        prompt = ANSWER_PROMPT.format(context=context, question=question)
        if self.llm is None:
            try:
                from RAG.src.LargeLanguageModel import LargeLanguageModel
                default_model = os.getenv("DEFAULT_ANSWER_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
                self.llm = LargeLanguageModel(default_model, use_fp16=False)
            except Exception as exc:
                raise RuntimeError("LLM is not available for answer generation") from exc

        answer = self.llm.decode(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        answer_text = answer.strip()
        cited_node_ids = []
        try:
            import re
            matches = re.findall(r"\bdoc(?:\.[0-9]+)+\b", answer_text)
            cited_node_ids = sorted(set(match for match in matches))
        except Exception:
            cited_node_ids = []

        return {
            "answer": answer_text,
            "cited_node_ids": cited_node_ids,
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(answer.split()),
        }
