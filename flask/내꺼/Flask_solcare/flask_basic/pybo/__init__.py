from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import config
db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    from . import models
    db.init_app(app)
    migrate.init_app(app, db)

    from .views import main_views, mci_views, auth_views, genai_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(mci_views.bp)
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(genai_views.bp)

    return app
