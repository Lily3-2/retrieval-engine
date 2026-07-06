from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class BaselineRetriever:
    def __init__(self, pdf_path: str, model_id: str = "gpt2", chunk_size: int = 200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        try:
            from RAG.src.LargeLanguageModel import LargeLanguageModel
            from RAG.src.IndexManager import IndexManager
            from RAG.src.PDFProcessor import PDFProcessor
        except ImportError as exc:
            raise RuntimeError("Unable to import RAG baseline dependencies") from exc

        self.llm = LargeLanguageModel(model_id, use_fp16=False)
        self.processor = PDFProcessor()
        texts, info = self._load_pdf_chunks(pdf_path)
        self.index_manager = IndexManager(texts, info, self.llm)
        self.index_manager.create_index(texts)

    def _load_pdf_chunks(self, pdf_path: str) -> tuple[List[str], List[str]]:
        if os.path.isdir(pdf_path):
            texts, info = self.processor.extract_text_from_pdfs_in_folder(pdf_path)
        else:
            folder = os.path.dirname(pdf_path)
            texts, info = self.processor.extract_text_from_pdfs_in_folder(folder)
            if os.path.basename(pdf_path) not in [itm.split()[0] for itm in info]:
                raise ValueError(f"PDF file not found in folder: {pdf_path}")
        from RAG.src.IndexManager import IndexManager as _IndexManager
        chunks, chunk_info = _IndexManager.chunk_texts_and_info(texts, info, self.chunk_size)
        return chunks, chunk_info

    def retrieve(self, question: str, top_k: int = 8) -> List[Dict[str, Any]]:
        results = self.index_manager.query(question, top_k=top_k)
        return [{"text": text, "info": info, "score": None} for text, info in results]
