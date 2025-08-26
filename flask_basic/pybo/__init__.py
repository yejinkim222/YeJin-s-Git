from flask import Flask
from dotenv import load_dotenv
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
import config
from pybo.agent.cli import register_cli

db = SQLAlchemy()      # ← 여기서 단 한 번만 생성
migrate = Migrate()

def create_app(config_name: str | None = None) -> Flask:
    load_dotenv()

<<<<<<< HEAD
    # 기본 설정 로드
    app.config.from_object("config")

    # 프로파일별 오버라이드 (선택)
    if config_name == "dev":
        app.config.update(DEBUG=True)
    elif config_name == "test":
        app.config.update(TESTING=True)

    # 확장 초기화
    # db.init_app(app); migrate.init_app(app, db)

    # 블루프린트 등록
    from .views.main_views import bp as main_bp
    from .views.genai_views import bp as genai_bp
    from .views.disease_views import bp as disease_bp  # 있다면
    app.register_blueprint(main_bp)
    app.register_blueprint(genai_bp)
    app.register_blueprint(disease_bp)

    # per-app 저장소(전역 lru_cache 대체)
=======
    app = Flask(__name__)
    app.config.from_object(config)
>>>>>>> a68f713665d4fa00dd57d6c94a8805c6a9c82018
    app.extensions.setdefault("llm_store", {})

    db.init_app(app)
    migrate.init_app(app, db)

    from . import models

    from .views import (
        main_views, disease_views, mci_views, auth_views, genai_views)
    app.register_blueprint(main_views.bp)
    app.register_blueprint(genai_views.bp)
    app.register_blueprint(disease_views.bp)
    app.register_blueprint(mci_views.bp)
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
