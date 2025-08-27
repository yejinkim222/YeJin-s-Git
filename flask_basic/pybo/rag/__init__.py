# pybo/rag/__init__.py

"""
RAG 패키지 공개 API (engine.py를 래핑).
"""
from .engine import (
    rag_upsert_pdfs,
    rag_sync_env_pdf_dir,
    rag_upsert_texts,
    rag_search_snippets,
)

__all__ = [
    "rag_upsert_pdfs",
    "rag_sync_env_pdf_dir",
    "rag_upsert_texts",
    "rag_search_snippets",
]
__version__ = "0.1.0"

