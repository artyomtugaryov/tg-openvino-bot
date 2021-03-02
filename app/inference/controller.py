from typing import List

from flask_accepts import responds
from flask_restx import Namespace, Resource

from app.inference.schema import InferenceSchema
from app.inference.service import InferenceService

api = Namespace('Inference')


@api.route('/')
class InferenceResource(Resource):

    @responds(schema=InferenceSchema)
    def get(self) -> List:
        return InferenceService().do()
