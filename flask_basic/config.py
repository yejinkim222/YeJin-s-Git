import os
from os import getenv

# 기본 설정
BASE_DIR = os.path.dirname(__file__)
SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR,'solcare.db'))
SQLALCHEMY_TRACK_MODIFICATIONS = False
# 암호 아직 제대로 설정된 것 아님
SECRET_KEY="dev"


# 샘플링
GENAI_TEMPERATURE = float(getenv("GENAI_TEMPERATURE", 0.2))
GENAI_TOP_P = float(getenv("GENAI_TOP_P", 0.85))
GENAI_MAX_NEW_TOKENS = int(getenv("GENAI_MAX_NEW_TOKENS", 320))
GENAI_REPETITION_PENALTY = float(getenv("GENAI_REPETITION_PENALTY", 1.1))
GENAI_NO_REPEAT_NGRAM_SIZE = int(getenv("GENAI_NO_REPEAT_NGRAM_SIZE", 4))
GENAI_MAX_CTX_MESSAGES = int(getenv("GENAI_MAX_CTX_MESSAGES", 16))
GENAI_MAX_MESSAGES = int(getenv("GENAI_MAX_MESSAGES", 80))


# ---- LLM 백엔드 선택 ----
GENAI_BACKEND = os.getenv("GENAI_BACKEND", "local")   # local | gemini | exaone

# ---- Gemini ----
# GEMINI_MODEL   = getenv("GEMINI_MODEL", "gemini-1.5-flash")
# GEMINI_API_KEY = getenv("GOOGLE_API_KEY", "")

# ---- OLLAMA(llama3) ----
OLLAMA_BASE_URL = getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL    = getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q5_K_M")
GENAI_PROVIDER = os.getenv("GENAI_PROVIDER", "ollama")   # ← 중요: 키 이름 교정

# ---- Whisper ----
DEMENTIA_UPLOAD_DIR = os.getenv("DEMENTIA_UPLOAD_DIR",
                                os.path.join(BASE_DIR, "data", "uploads", "dementia"))
DEMENTIA_ASR_MODEL_SIZE = os.getenv("DEMENTIA_ASR_MODEL_SIZE", "small")
DEMENTIA_CLF_PKL = os.getenv("DEMENTIA_CLF_PKL",
                             os.path.join(BASE_DIR, "python", "dmt_model", "clf.pkl"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))

ASR_MODEL = os.getenv("ASR_MODEL", "small")   # tiny/base/small/medium/large-v3 등
ASR_BEAM  = int(os.getenv("ASR_BEAM", 5))


# RAG
RAG_ENABLED = (str(getenv("RAG_ENABLED", "True")).lower() == "true")
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 3))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 1000))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 100))
RAG_DIR = os.path.join(BASE_DIR, "data", "rag")
RAG_INDEX_PATH = getenv("RAG_INDEX_PATH", os.path.join(RAG_DIR, "faiss.index"))
RAG_META_PATH  = getenv("RAG_META_PATH",  os.path.join(RAG_DIR, "meta.json"))
RAG_CHUNK_PREVIEW_N = int(os.getenv("RAG_CHUNK_PREVIEW_N", 100))

RAG_ALLOW_HOSTS = [s.strip() for s in getenv("RAG_ALLOW_HOSTS","").split(",") if s.strip()]
RAG_SEED_URLS = [s.strip() for s in getenv("RAG_SEED_URLS","").split(",") if s.strip()]
RAG_CRAWL_DEPTH = int(getenv("RAG_CRAWL_DEPTH", 1))
RAG_MAX_PAGES_PER_HOST = int(getenv("RAG_MAX_PAGES_PER_HOST", 20))
RAG_RESPECT_ROBOTS = str(getenv("RAG_RESPECT_ROBOTS","true")).lower()=="true"
RAG_RATE_LIMIT_PER_HOST = float(getenv("RAG_RATE_LIMIT_PER_HOST", 0.5))

# --- RAG / Chroma 기본값 & ENV 매핑 ---
RAG_PDF_DIR = os.getenv("RAG_PDF_DIR", "")  # 비워두면 동기화 스킵
RAG_CHROMA_DIR = os.getenv(
    "RAG_CHROMA_DIR",
    os.path.join(BASE_DIR, "data", "rag", "chroma")  # 기본 경로
)

# bot 토글
BOT_RAG_MODE   = getenv("BOT_RAG_MODE", "false")
BOT_AGENT_MODE = getenv("BOT_AGENT_MODE", "false")
BOT_MAX_DISPLAY_CHARS = 500         # 일반 챗 표시 길이
BOT_MAX_DISPLAY_CHARS_RAG = 900     # RAG/에이전트 표시 길이