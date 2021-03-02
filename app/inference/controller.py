from flask import request, send_file
from flask_restx import Namespace, Resource

from app.inference.service import InputData, HorseInferenceService

api = Namespace('Inference')


@api.route('/horse', methods=['POST'])
class InferenceResource(Resource):

    def post(self):
        request_data = InputData(request)
        inference_result = HorseInferenceService().infer(request_data)
        response_data = inference_result.prepare_to_send()
        return send_file(response_data, mimetype='image/PNG')
