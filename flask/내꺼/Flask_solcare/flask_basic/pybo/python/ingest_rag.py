from __future__ import annotations
import os, json, re
from typing import List, Dict
from pathlib import Path

from flask import Flask
from pybo.rag.retriever import get_retriever

from langchain_text_splitters import RecursiveCharacterTextSplitter



def chunk_text(text: str, max_chars=450, overlap=80) -> list[str]:
    """
    LangChain의 RecursiveCharacterTextSplitter를 이용해 텍스트를 슬라이딩 청킹한다.
    입력 텍스트의 여분 공백을 정리한 뒤 split_text 사용.
    """
    text = re.sub(r"\s+\n", "\n", text).strip()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        # 문단/줄/공백 우선 분할
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def load_plain_docs(root: str) -> List[Dict[str, str]]:
    """root 아래의 .txt/.md를 읽어 {text, source} 리스트로."""
    docs: List[Dict[str, str]] = []
    for path in Path(root).rglob("*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        txt = path.read_text(encoding="utf-8", errors="ignore")
        for i, ch in enumerate(chunk_text(txt)):
            docs.append({"text": ch, "source": f"{path.name}#p{i+1}"})
    return docs

def main():
    # Flask 컨텍스트( config 접근용 )
    app = Flask(__name__)
    app.config.from_object("pybo.config")  # 네 프로젝트의 config.py
    with app.app_context():
        retriever = get_retriever()
        docs_root = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
        docs_root = os.path.abspath(docs_root)
        os.makedirs(docs_root, exist_ok=True)

        passages = load_plain_docs(docs_root)
        if not passages:
            print(f"[ingest] {docs_root} 에 문서(.txt/.md)가 없습니다.")
            return
        print(f"[ingest] passages: {len(passages)}개 인덱싱 중...")
        retriever.add_passages(passages, persist=True)
        print("[ingest] 완료")

if __name__ == "__main__":
    main()
