# pybo/rag/__init__.py (lazy)
"""
RAG 패키지 공개 API (지연 임포트).
"""

__all__ = [
    "rag_crawl_and_index",
    "rag_upsert_texts",
    "rag_search_snippets",
]
__version__ = "0.1.0"

def rag_crawl_and_index(*args, **kwargs):
    from rag import rag_crawl_and_index as _impl
    return _impl(*args, **kwargs)

def rag_upsert_texts(*args, **kwargs):
    from rag import rag_upsert_texts as _impl
    return _impl(*args, **kwargs)

def rag_search_snippets(*args, **kwargs):
    from rag import rag_search_snippets as _impl
    return _impl(*args, **kwargs)
