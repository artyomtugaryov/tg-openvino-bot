from flask_restx.api import Api


def register_routes(api: Api, root: str):
    base_route = 'inference'

    from .controller import api as inference_api
    # localhost:500/api/inference

    api.add_namespace(inference_api, path=f'/{root}/{base_route}')

