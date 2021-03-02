from flask_restx.api import Api


def register_routes(api: Api, root: str = 'api'):
    from app.inference import register_routes as attach_inference

    # Add routes
    attach_inference(api=api, root=root)
