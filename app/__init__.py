from flask import Flask
from flask_restx import Api


def create_app(env: str = 'test') -> Flask:
    from app.config import get_config
    from app.routes import register_routes

    app = Flask(__name__)
    app.config.from_object(get_config(env))
    api = Api(app, title='CycleGan in OpenVINO', version="0.1.0")

    register_routes(api)

    return app
