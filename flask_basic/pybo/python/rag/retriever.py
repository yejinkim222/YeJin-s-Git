# pybo/rag/retriever.py
from __future__ import annotations
import os, json, math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Optional, Tuple

import numpy as np
from flask import current_app

try:
    import faiss  # faiss-cpu 권장

    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "sentence-transformers가 필요합니다: pip install sentence-transformers"
    ) from e


def _normalize(a: np.ndarray) -> np.ndarray:
    """L2 정규화."""
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / n


@dataclass
class Passage:
    text: str
    source: str
    score: float


class _BaseIndex:
    def add(self, vecs: np.ndarray): ...
    def search(self, q: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]: ...
    def save(self, path: str): ...
    @classmethod
    def load(cls, path: str, dim: int) -> "_BaseIndex": ...


class _FaissIPIndex(_BaseIndex):
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vecs: np.ndarray):
        self.index.add(vecs)

    def search(self, q: np.ndarray, top_k: int):
        return self.index.search(q, top_k)

    def save(self, path: str):
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, dim: int):
        idx = cls(dim)
        idx.index = faiss.read_index(path)
        return idx


class _NumpyIPIndex(_BaseIndex):
    """FAISS가 없을 때를 위한 아주 단순한 대체."""

    def __init__(self, dim: int):
        self.vecs = np.zeros((0, dim), dtype=np.float32)

    def add(self, vecs: np.ndarray):
        self.vecs = np.vstack([self.vecs, vecs]) if self.vecs.size else vecs

    def search(self, q: np.ndarray, top_k: int):
        sims = q @ self.vecs.T  # (1, N)
        idx = np.argsort(-sims, axis=1)[:, :top_k]
        val = np.take_along_axis(sims, idx, axis=1)
        return val, idx

    def save(self, path: str):
        np.save(path, self.vecs)

    @classmethod
    def load(cls, path: str, dim: int):
        idx = cls(dim)
        idx.vecs = np.load(path + ".npy")
        return idx


class RagRetriever:
    """
    텍스트 임베딩 + (FAISS or Numpy) 내적 검색기.
    meta.json에는 각 벡터의 {text, source} 메타데이터가 순서대로 저장됨.
    """

    def __init__(self, model_name: str, index_path: str, meta_path: str):
        self.model_name = model_name
        self.index_path = index_path
        self.meta_path = meta_path
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index: _BaseIndex = (
            _FaissIPIndex(self.dim) if _HAS_FAISS else _NumpyIPIndex(self.dim)
        )
        self.meta: List[Dict] = []

        # 로드(있으면)
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self._load()

    # ---------- 저장/로드 ----------
    def _load(self):
        if _HAS_FAISS:
            self.index = _FaissIPIndex.load(self.index_path, self.dim)
        else:
            self.index = _NumpyIPIndex.load(self.index_path, self.dim)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        if _HAS_FAISS:
            self.index.save(self.index_path)
        else:
            self.index.save(self.index_path)  # .npy로 저장됨
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    # ---------- 인덱싱 ----------
    def add_passages(self, passages: List[Dict[str, str]], persist: bool = True):
        """
        passages: [{"text": "...", "source": "파일/섹션"}...]
        """
        texts = [p["text"] for p in passages]
        vecs = self.embed(texts, is_query=False)
        self.index.add(vecs)
        self.meta.extend(passages)
        if persist:
            self._save()

    def embed(self, texts: List[str], is_query: bool) -> np.ndarray:
        """
        BGE-M3는 쿼리와 문서에 약한 프롬프트 차이가 있지만, 우선 공통 임베딩으로도 충분히 동작.
        필요시 여기서 is_query=True일 때 프롬프트/정규화 로직을 살짝 다르게 해도 됨.
        """
        embs = self.embedder.encode(
            texts, batch_size=32, normalize_embeddings=False, convert_to_numpy=True
        )
        return _normalize(embs)

    # ---------- 검색 ----------
    def search(self, query: str, top_k: int = 3) -> List[Passage]:
        qv = self.embed([query], is_query=True)  # (1, d)
        scores, idxs = self.index.search(qv, top_k)
        hits: List[Passage] = []
        for i, sc in zip(idxs[0], scores[0]):
            if i < 0 or i >= len(self.meta):
                continue
            m = self.meta[i]
            hits.append(
                Passage(text=m["text"], source=m.get("source", ""), score=float(sc))
            )
        return hits


# --------- 애플리케이션에서 쉽게 쓰는 진입점 ---------
@lru_cache(maxsize=1)
def get_retriever() -> RagRetriever:
    cfg = current_app.config
    return RagRetriever(
        model_name=cfg.get("RAG_EMBED_MODEL", "BAAI/bge-m3"),
        index_path=cfg.get("RAG_INDEX_PATH"),
        meta_path=cfg.get("RAG_META_PATH"),
    )
