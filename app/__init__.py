from flask import Flask
from app.extensions import db, migrate
from app.routes.main import main_bp
from app.routes.auth import auth_bp
from app.routes.auth_endpoint import auth_ep
from app.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # Register Blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(auth_ep)

    return app
