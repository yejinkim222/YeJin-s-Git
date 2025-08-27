from flask import Flask
from dotenv import load_dotenv
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
import config
from pybo.agent.cli import register_cli
from langchain import requests


db = SQLAlchemy()
migrate = Migrate()

# 빠른 로딩 위해 미리 불러오기
def _ollama_warmup(app):
    base = (app.config.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    model = app.config.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q5_K_M")
    keep_alive = app.config.get("OLLAMA_KEEP_ALIVE", "30m")
    timeout_health = int(app.config.get("LLM_WARMUP_HEALTH_TIMEOUT", 3))
    timeout_chat   = int(app.config.get("LLM_WARMUP_CHAT_TIMEOUT", 20))


    try:
        # 떴는지 확인
        requests.get(f"{base}/v1/models", timeout=timeout_health)

        # 모델 메모리 올리기
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 1,
            "options": {"num_predict": 1},
            "keep_alive": keep_alive,
        }
        requests.post(f"{base}/v1/chat/completions", json=payload, timeout=timeout_chat)
        app.logger.info("LLM warm-up done")
    except Exception as e:
        app.logger.warning("LLM warm-up skipped: %s", e)


def create_app(config_name: str | None = None) -> Flask:
    load_dotenv()

    app = Flask(__name__)
    app.config.from_object(config)
    app.extensions.setdefault("llm_store", {})

    db.init_app(app)
    migrate.init_app(app, db)

    from . import models

    from .views import (
        main_views, disease_views, mci_views, genai_views, auth_views)
    app.register_blueprint(main_views.bp)
    app.register_blueprint(disease_views.bp)
    app.register_blueprint(mci_views.bp)
    app.register_blueprint(genai_views.bp)
    app.register_blueprint(auth_views.bp)

    with app.app_context():
        try:
            from pybo.views.genai_views import get_llm
            llm = get_llm()
            _ = llm.chat([{"role": "user", "content": "ok"}], max_new_tokens=1)
            app.logger.info("LLM warm-up done")
        except Exception as e:
            app.logger.warning("LLM warm-up skipped: %s", e)

    register_cli(app)

    return app
