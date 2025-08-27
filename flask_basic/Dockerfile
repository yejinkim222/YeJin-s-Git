# ---- Build stage (옵션: 빌드 캐시/의존성 분리하고 싶으면 유지) ----
FROM python:3.11-slim AS build
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --upgrade pip && pip wheel --wheel-dir /wheels -r requirements.txt

# ---- Run stage ----
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# 런타임에 필요한 패키지만
RUN apt-get update && apt-get install -y libstdc++6 && rm -rf /var/lib/apt/lists/*

# 의존성/소스 복사
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && rm -rf /wheels
COPY . .

# Cloud Run은 $PORT로 넘겨줌 → gunicorn이 이 포트 바인딩
ENV PORT=8080
# Chroma / SQLite / 임시파일은 /tmp 사용(Cloud Run에서 유일한 쓰기 가능한 경로, 영속 X)
ENV RAG_CHROMA_DIR=/tmp/chroma
ENV SQLALCHEMY_DATABASE_URI=sqlite:////tmp/solcare.db

# gunicorn: 애플리케이션 팩토리 사용 (pybo:create_app())
# 타임아웃/스레드/프록시 헤더 처리
CMD exec gunicorn 'pybo:create_app()' \
  --bind ":${PORT}" \
  --workers 2 \
  --threads 8 \
  --timeout 180 \
  --access-logfile '-' \
  --error-logfile '-' \
  --forwarded-allow-ips='*'
