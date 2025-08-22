from flask import Flask
from dotenv import load_dotenv
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()      # ← 여기서 단 한 번만 생성
migrate = Migrate()

def create_app(config_name: str | None = None) -> Flask:
    """앱 인스턴스를 매번 새로 만들어 반환한다."""
    load_dotenv()  # .env 자동 로드(프로세스 단위지만, 여기서 한 번만)
    app = Flask(__name__, instance_relative_config=True)

    # 기본 설정 로드
    app.config.from_object("pybo.config")

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
    app.extensions.setdefault("llm_store", {})
    app.extensions.setdefault("asr_store", {})
    app.extensions.setdefault("rag_store", {})

    return app
