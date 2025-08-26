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

    app = Flask(__name__)
    app.config.from_object(config)
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
