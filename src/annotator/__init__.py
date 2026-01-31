from flask import Flask

from annotator.models import db
from annotator.routes import main_bp


def create_app():
    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite:///annotations.db",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)
    app.register_blueprint(main_bp)
    with app.app_context():
        db.create_all()
    return app
