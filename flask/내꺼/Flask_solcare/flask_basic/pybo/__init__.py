from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import config
db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    #ORM Initialization
    db.init_app(app)
    migrate.init_app(app, db)
    from . import models

    #Blueprint
    from .views import main_views, mci_input_views, mci_output_views, auth_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(input_views.bp)
    app.register_blueprint(output_views.bp)
    app.register_blueprint(auth_views.bp)
    return app
