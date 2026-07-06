ANSWER_PROMPT = """You are answering strictly from the provided structured evidence.
Each evidence block is labeled with its node_id and semantic path.
Answer concisely. After your answer, cite the node_id(s) you used.
If the evidence does not contain the answer, say "Not found in document.".

Evidence:
{context}

Question: {question}
Answer:"""


def format_context_block(node_id: str, semantic_path: list[str], text: str, physical_location: dict) -> str:
    path = " > ".join(semantic_path) if semantic_path else ""
    location = f" [location={physical_location}]" if physical_location else ""
    return f"[{node_id}] {path}{location}\n{text.strip()}"
