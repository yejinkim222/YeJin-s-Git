# pybo/rag/engine.py
from __future__ import annotations

import os, re, json, hashlib
from dataclasses import dataclass
from typing import List, Dict
from flask import current_app


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document  # fallback
try:
    from langchain.embeddings.base import Embeddings
except Exception:
    from langchain_core.embeddings import Embeddings  # fallback

from langchain_chroma import Chroma
import chromadb
from FlagEmbedding import BGEM3FlagModel


# =========================
# 임베딩 어댑터(BGE-M3)
# =========================
class _BgeM3Embeddings(Embeddings):
    """FlagEmbedding BGEM3 -> LangChain Embeddings 어댑터"""
    def __init__(self, model_id: str):
        self.model = BGEM3FlagModel(model_id, use_fp16=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, batch_size=64)["dense_vecs"]
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        v = self.model.encode([text])["dense_vecs"][0]
        return v.tolist()


def _get_embedder() -> Embeddings:
    store = current_app.extensions.setdefault("rag_store", {})
    if "embedder" not in store:
        model_id = current_app.config.get("RAG_EMBED_MODEL", "BAAI/bge-m3")
        store["embedder"] = _BgeM3Embeddings(model_id)
    return store["embedder"]


def _get_chroma() -> Chroma:
    """
    chroma DB 영속 디렉터리 사용(표준 방식). persist() 호출 불필요/미사용.
    """
    store = current_app.extensions.setdefault("rag_store", {})
    if "chroma" not in store:
        persist_dir = current_app.config.get("RAG_CHROMA_DIR", "./data/chroma")
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        collection = current_app.config.get("RAG_CHROMA_COLLECTION", "solcare_rag")
        store["chroma"] = Chroma(
            client=client,
            collection_name=collection,
            embedding_function=_get_embedder(),
        )
    return store["chroma"]


# =========================
# 유틸: 제목/문서키/청킹
# =========================
@dataclass
class PdfMeta:
    title: str
    title_is_guess: bool
    pages_total: int
    file_path: str


def _guess_pdf_title(pdf_path: str) -> PdfMeta:
    """
    PyPDFLoader만 사용.
    1쪽 텍스트 상단 30줄에서 제목 후보 추정 → 없으면 파일명.
    """
    title, title_is_guess, pages_total = "", True, 0
    try:
        pages = PyPDFLoader(pdf_path).load()  # List[Document]
        pages_total = len(pages)
        if pages:
            head = (pages[0].page_content or "").splitlines()
            head = [re.sub(r"\s+", " ", ln.strip()) for ln in head[:30]]
            head = [ln for ln in head if ln]
            bad = re.compile(r"^(abstract|요약|목차|contents|서론|introduction)\b", re.I)
            cand = [ln for ln in head if len(ln) >= 6 and not bad.match(ln)]
            good = [ln for ln in cand if 8 <= len(ln) <= 120]
            picked = (max(good, key=len) if good else (max(cand, key=len) if cand else ""))
            if picked:
                title = picked if len(picked) <= 120 else picked[:120].rstrip() + "…"
                title_is_guess = True
    except Exception:
        current_app.logger.warning("RAG: title guess via PyPDFLoader failed: %s", pdf_path)

    if not title:
        title, title_is_guess = os.path.splitext(os.path.basename(pdf_path))[0], True
    return PdfMeta(title=title, title_is_guess=title_is_guess, pages_total=pages_total, file_path=pdf_path)


def _make_splitter() -> RecursiveCharacterTextSplitter:
    size = int(current_app.config.get("RAG_CHUNK_SIZE", 1000))
    overlap = int(current_app.config.get("RAG_CHUNK_OVERLAP", 100))
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)


def _attach_preview(chunks: List[str]) -> List[str]:
    """다음 청크의 앞부분을 붙이는 look-ahead preview."""
    preview_n = int(current_app.config.get("RAG_CHUNK_PREVIEW_N", 100))
    if preview_n <= 0 or len(chunks) <= 1:
        return chunks
    out = []
    for i in range(len(chunks) - 1):
        out.append(f"{chunks[i]}\n{chunks[i+1][:preview_n]}")
    out.append(chunks[-1])
    return out

# --- doc_key: 경로/내용 전략 분리 ---

def _doc_key_path(path: str) -> str:
    ap = os.path.abspath(path).replace("\\", "/").lower()
    return "path:" + hashlib.sha1(ap.encode("utf-8", "ignore")).hexdigest()

_DOI_RE = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"

def _extract_doi(text: str) -> str | None:
    m = re.search(_DOI_RE, text, flags=re.I)
    return m.group(0).lower() if m else None

def _norm_title(s: str) -> str:
    s = (s or "").lower()
    return re.sub(r"[\s\[\]\(\)\{\}:;,.]+", " ", s).strip()

def _doc_key_text(title: str, raw_text: str) -> str:
    doi = _extract_doi(raw_text or "")
    if doi:
        return f"doi:{doi}"
    base = _norm_title(title)
    if base:
        return "t:" + hashlib.sha1(base.encode("utf-8", "ignore")).hexdigest()
    return "h:" + hashlib.sha1((raw_text or "").encode("utf-8", "ignore")).hexdigest()

def _choose_doc_key(path: str | None = None, title: str = "", text: str = "") -> str:
    """
    PDF에는 보통 path 전략을, 텍스트 문서에는 content 전략을 쓰는 게 합리적입니다.
    RAG_DOC_KEY_STRATEGY=path|content (기본: path)
    """
    strategy = str(current_app.config.get("RAG_DOC_KEY_STRATEGY", "path")).lower()
    if strategy == "content":
        # 내용 우선
        return _doc_key_text(title, text)
    # 기본: 경로 우선(경로 없으면 내용으로 폴백)
    if path:
        return _doc_key_path(path)
    return _doc_key_text(title, text)

# 하나만 남기세요: 리스트형 프리뷰 부착기
from typing import List

def _chunks_with_preview(splits: List[str]) -> List[str]:
    """splitter로 자른 청크 리스트에 '다음 청크의 앞 preview_n자'를 붙여준다."""
    preview_n = int(current_app.config.get("RAG_CHUNK_PREVIEW_N", 100))
    # 공백 제거 & 빈 청크 제거
    splits = [s.strip() for s in (splits or []) if s and s.strip()]
    if len(splits) <= 1 or preview_n <= 0:
        return splits

    out: List[str] = []
    for i, cur in enumerate(splits):
        if i < len(splits) - 1:
            nxt = splits[i + 1][:preview_n]
            out.append(f"{cur}\n{nxt}")
        else:
            out.append(cur)
    return out


# =========================
# 파일 레지스트리(폴더 동기화용)
# =========================
def _registry_path() -> str:
    rag_dir = os.path.join(os.getcwd(), "pybo", "data", "rag")
    os.makedirs(rag_dir, exist_ok=True)
    return os.path.join(rag_dir, "ingested_pdfs.json")


def _load_registry() -> Dict[str, Dict]:
    try:
        with open(_registry_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_registry(reg: Dict[str, Dict]):
    with open(_registry_path(), "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================
# 1) PDF 업서트 (append: 기존 것은 삭제하지 않지만,
#    같은 파일 경로의 이전 청크는 교체 위해 where={"file_path": path}로 제거)
# =========================
def rag_upsert_pdfs(paths: List[str]) -> Dict:
    """
    지정한 PDF들을 파싱 → 페이지별 split → preview 부착 → 중복 필터(doc_key/청크 해시) →
    동일 doc_key 선삭제 후 add_documents로 교체(업서트).
    ※ persist()는 호출하지 않습니다.
    """
    added = 0
    skipped, errors = [], []
    vectordb = _get_chroma()
    splitter = _make_splitter()

    for p in (paths or []):
        try:
            if not os.path.exists(p) or not p.lower().endswith(".pdf"):
                skipped.append({"path": p, "reason": "file_not_found_or_not_pdf"})
                current_app.logger.warning("RAG: skip (not found/pdf): %s", p)
                continue

            meta = _guess_pdf_title(p)
            loader = PyPDFLoader(p)
            pages = loader.load()
            if not pages:
                skipped.append({"path": p, "reason": "empty_pdf"})
                current_app.logger.warning("RAG: empty pdf: %s", p)
                continue

            total_text = "".join([(d.page_content or "") for d in pages])
            if len(total_text.strip()) < 200:
                # 스캔/보호 PDF 의심
                skipped.append({"path": p, "reason": "too_short_or_scanned"})
                current_app.logger.warning("RAG: too short / scanned? %s", p)
                continue

            dockey = _choose_doc_key(path=p, title=meta.title, text=total_text)

            # 업서트 전 삭제
            try:
                vectordb.delete(where={"doc_key": dockey})
            except Exception:
                pass

            # 같은 파일 경로로 들어온 과거 청크 제거
            try:
                vectordb.delete(where={"file_path": os.path.abspath(p)})
            except Exception:
                pass

            docs_to_add: List[Document] = []
            for d in pages:
                page_num = int(d.metadata.get("page", 0)) + 1  # 1-indexed
                raw = (d.page_content or "").strip()
                if not raw:
                    continue

                splits = splitter.split_text(raw)
                splits = _chunks_with_preview(splits)

                # 동일 문서 내 중복 청크 제거
                seen_hash = set()
                for chunk in splits:
                    hh = hashlib.sha1(chunk.encode("utf-8", "ignore")).hexdigest()
                    if hh in seen_hash:
                        continue
                    seen_hash.add(hh)

                    docs_to_add.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "title": meta.title if not meta.title_is_guess else f"[추정] {meta.title}",
                                "title_is_guess": bool(meta.title_is_guess),
                                "file_path": meta.file_path,
                                "pages_total": meta.pages_total,
                                "page_from": page_num,
                                "page_to": page_num,
                                "doc_key": dockey,
                                "source": f"file://{meta.file_path}#page={page_num}",
                            },
                        )
                    )

            if not docs_to_add:
                skipped.append({"path": p, "reason": "no_chunks"})
                continue

            vectordb.add_documents(docs_to_add)
            added += len(docs_to_add)

        except Exception as e:
            errors.append({"path": p, "error": str(e)})
            current_app.logger.exception("RAG: pdf upsert error: %s", p)

    return {"ok": True, "added": added, "skipped": skipped, "errors": errors}


# =========================
# 2) 환경변수 폴더 동기화(새/변경 PDF만 추가)
# =========================
def rag_sync_env_pdf_dir() -> Dict:
    base = current_app.config.get("RAG_PDF_DIR")
    if not base:
        return {"ok": False, "msg": "RAG_PDF_DIR not set"}
    if not os.path.isdir(base):
        return {"ok": False, "msg": f"RAG_PDF_DIR not found: {base}"}

    pdfs = [os.path.join(base, n) for n in os.listdir(base) if n.lower().endswith(".pdf")]
    reg = _load_registry()

    targets = []
    for p in pdfs:
        ap = os.path.abspath(p)
        try:
            sha1 = _file_sha1(ap)
        except Exception:
            current_app.logger.warning("RAG: cannot read file: %s", ap)
            continue
        if (reg.get(ap) or {}).get("sha1") != sha1:
            targets.append(ap)

    if not targets:
        return {"ok": True, "msg": "no new/changed pdf", "added": 0}

    return rag_upsert_pdfs(targets)


# =========================
# 3) 내부 텍스트 업서트(지침/FAQ)
# =========================
def rag_upsert_texts(items: List[Dict]) -> Dict:
    """
    내부 지침/FAQ 업서트. 제목(title) 기반 안정적 doc_key 사용.
    같은 제목이면 기존 문서를 delete(where={"doc_key": dockey}) 후 add_documents로 교체.
    """
    added = 0
    skipped, errors = [], []
    vectordb = _get_chroma()
    splitter = _make_splitter()

    for it in (items or []):
        try:
            title = (it.get("title") or "internal").strip()
            extra = it.get("meta") or {}
            text = (it.get("text") or "").strip()
            if len(text) < 50:
                skipped.append({"title": title, "reason": "too_short"})
                continue

            # 제목 기반 안정적 doc_key
            dockey = _doc_key_text(title, text)

            # 교체 전략: 같은 doc_key 전부 삭제
            try:
                vectordb.delete(where={"doc_key": dockey})
            except Exception:
                pass

            extra = it.get("meta") or {}
            splits = splitter.split_text(text)
            splits = _chunks_with_preview(splits)

            seen_hash = set()
            to_add: List[Document] = []
            for chunk in splits:
                hh = hashlib.sha1(chunk.encode("utf-8", "ignore")).hexdigest()
                if hh in seen_hash:
                    continue
                seen_hash.add(hh)

                to_add.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "title": title,
                            "title_is_guess": False,
                            "file_path": extra.get("file_path", "internal://guideline"),
                            "pages_total": extra.get("pages_total", 1),
                            "page_from": extra.get("page_from", 1),
                            "page_to": extra.get("page_to", 1),
                            "doc_key": dockey,
                            "source": extra.get("source", "internal://guideline"),
                            **extra,
                        },
                    )
                )

            if not to_add:
                skipped.append({"title": title, "reason": "no_chunks"})
                continue

            vectordb.add_documents(to_add)
            added += len(to_add)

        except Exception as e:
            errors.append({"title": it.get("title"), "error": str(e)})
            current_app.logger.exception("RAG: upsert_texts error: %s", it)

    # 여기서는 persist() 호출 안 함 (요청사항 반영)
    return {"ok": True, "added": added, "skipped": skipped, "errors": errors}


# =========================
# 4) 검색(Top-K 스니펫)
# =========================
def rag_search_snippets(query: str, top_k: int = 3) -> str:
    """
    질의어로 유사 청크 상위 K개를 찾아
    '짧은 스니펫 + 출처 힌트 + 키워드'로 반환.
    (스니펫은 가독성 위해 정리하고 200자로 제한)
    """
    try:
        vectordb = _get_chroma()
        results = vectordb.similarity_search_with_score(query, k=int(top_k))
    except Exception as e:
        current_app.logger.exception("RAG: search failed")
        return f"[error] RAG search failed: {e}"

    # 영어 불용어(간단 셋)
    _EN_STOP = {
        "the","and","for","with","that","this","from","have","has","had","are","was","were","be","been",
        "of","to","in","on","at","by","as","an","or","but","not","no","yes","it","its","is","if","into",
        "can","may","might","should","could","would","will","than","then","so","such","more","most","any",
        "all","some","many","much","also","about","over","under","between","within","without"
    }
    # 한국어 불용어(간단 셋)
    _KO_STOP = {
        "그리고","그러나","대한","관련","사용","있습니다","합니다","입니다","있는","위해","최근","포함",
        "등","및","에서","으로","하는","하여","했다","된다","수","때","대해","까지","같은","경우"
    }

    def _kw(text: str, n: int = 5) -> str:
        # 3~20자 한/영/숫자 토큰, 불용어 제외 후 상위 n개
        toks = re.findall(r"[A-Za-z가-힣0-9]{3,20}", (text or "").lower())
        stop = _EN_STOP | _KO_STOP
        freq = {}
        for t in toks:
            if t in stop:
                continue
            freq[t] = freq.get(t, 0) + 1
        return ", ".join([w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:n]])

    def _clean_snippet(s: str, limit: int = 200) -> str:
        # 개행/불릿/중복공백 정리 후 200자 제한
        s = re.sub(r"[\r\n]+", " ", s or "")
        s = re.sub(r"^\s*[\u2022•·◦●]+\s*", "", s)  # 앞머리 불릿 제거
        s = re.sub(r"\s{2,}", " ", s).strip()
        if len(s) > limit:
            s = s[:limit].rstrip() + "…"
        return s

    hints = []
    for doc, _score in results:
        chunk = doc.page_content or ""
        m = doc.metadata or {}
        title = m.get("title") or m.get("host") or "문서"
        page_from = m.get("page_from") or m.get("page") or 1
        src = m.get("source") or m.get("file_path") or ""
        snippet = _clean_snippet(chunk, limit=200)
        kws = _kw(chunk, n=5)

        hints.append(
            f"- {title} — {snippet}\n"
            f"  [출처] {title} · p.{page_from} · {src}\n"
            f"  [키워드] {kws}"
        )

    if not hints:
        return "색인된 자료에서 유의미한 결과를 찾지 못했습니다. 다른 표현으로 다시 물어봐 주세요."
    return "다음 자료가 유용할 수 있어요. 더 자세히 보시겠다면 특정 항목을 지목해 주세요:\n" + "\n".join(hints)
