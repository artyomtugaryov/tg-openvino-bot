from flask import Flask
from flask_restx import Api


def create_app() -> Flask:
    from app.config import get_config
    from app.routes import register_routes

    app = Flask(__name__)
    config = get_config()
    app.config.from_object(config)
    api = Api(app, title='CycleGan in OpenVINO', version='0.1.0')

    register_routes(api)

    return app
