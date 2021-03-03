from flask import request, send_file
from flask_restx import Namespace, Resource

from app.inference.input_data import GenderTransformerInputData
from app.inference.service import GenderInferenceService

api = Namespace('Inference')


@api.route('/gender', methods=['POST'])
class InferenceResource(Resource):

    def post(self):
        request_data = GenderTransformerInputData(request).using(GenderInferenceService).infer().prepare_to_send()
        return send_file(request_data, mimetype='image/PNG')
