from __future__ import annotations

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
                self.llm = LargeLanguageModel("gpt2", use_fp16=False)
            except Exception as exc:
                raise RuntimeError("LLM is not available for answer generation") from exc

        answer = self.llm.decode(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return {
            "answer": answer.strip(),
            "cited_node_ids": [],
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(answer.split()),
        }
