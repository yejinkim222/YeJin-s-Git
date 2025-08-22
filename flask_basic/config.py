import os
from os import getenv

# 기본 설정
BASE_DIR = os.path.dirname(__file__)
print("BASE_DIR:",BASE_DIR)

SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR,'solcare.db'))

print("SQLALCHEMY_DATABASE_URI:",SQLALCHEMY_DATABASE_URI)
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 시크릿 키는 암호 만들려면 해야하는데 아직 안함
SECRET_KEY="dev"



# 모델 설정
# --- 모델 프리셋 또는 직접 HF 경로 ---
GENAI_MODEL = os.getenv("GENAI_MODEL", "qwen2.5-3b")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "")

# 샘플링(보수적)
GENAI_TEMPERATURE = float(os.getenv("GENAI_TEMPERATURE", 0.2))
GENAI_TOP_P = float(os.getenv("GENAI_TOP_P", 0.85))
GENAI_MAX_NEW_TOKENS = int(os.getenv("GENAI_MAX_NEW_TOKENS", 200))
GENAI_REPETITION_PENALTY = float(os.getenv("GENAI_REPETITION_PENALTY", 1.1))
GENAI_NO_REPEAT_NGRAM_SIZE = int(os.getenv("GENAI_NO_REPEAT_NGRAM_SIZE", 4))
GENAI_MAX_CTX_MESSAGES = int(os.getenv("GENAI_MAX_CTX_MESSAGES", 16))
GENAI_MAX_MESSAGES = int(os.getenv("GENAI_MAX_MESSAGES", 80))


# ---- LLM 백엔드 선택 ----
GENAI_BACKEND = os.getenv("GENAI_BACKEND", "local")   # local | gemini | exaone

# ---- Gemini ----
GENAI_PROVIDER = os.getenv("GENAI_PROVIDER", "gemini")  # 'gemini' | 'local'
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")  # 또는 'gemini-1.5-flash'
# API 키는 환경변수 GOOGLE_API_KEY 를 권장. (없으면 아래에서 current_app.config['GEMINI_API_KEY'] 참조)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ---- EXAONE ----
EXAONE_API_KEY  = os.getenv("EXAONE_API_KEY", "")
EXAONE_API_BASE = os.getenv("EXAONE_API_BASE", "https://api.exaone.ai/v1")  # 공급자 문서에 맞게 수정
EXAONE_MODEL    = os.getenv("EXAONE_MODEL", "exaone-3.0-instruct")          # 실제 모델 id로 수정

# ---- Whisper ----
DEMENTIA_ASR_MODEL_SIZE = os.getenv("DEMENTIA_ASR_MODEL_SIZE", "medium")  # tiny/base/small/medium/large-v3
DEMENTIA_CLF_PKL = os.getenv("DEMENTIA_CLF_PKL", os.path.join(BASE_DIR, "data", "models", "dementia_clf.pkl"))
DEMENTIA_UPLOAD_DIR = os.getenv("DEMENTIA_UPLOAD_DIR", os.path.join(BASE_DIR, "data", "uploads"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))

ASR_MODEL = os.getenv("ASR_MODEL", "small")   # tiny/base/small/medium/large-v3 등
ASR_BEAM  = int(os.getenv("ASR_BEAM", 5))


# RAG 관련
# RAG 기본 설정
RAG_ENABLED = True
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")  # 멀티링구얼 추천
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 3))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 450))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 80))

# 인덱스 저장 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, "data", "rag")
RAG_INDEX_PATH = os.getenv("RAG_INDEX_PATH", os.path.join(RAG_DIR, "faiss.index"))
RAG_META_PATH  = os.getenv("RAG_META_PATH",  os.path.join(RAG_DIR, "meta.json"))
