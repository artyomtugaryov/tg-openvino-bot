from flask import request, send_file
from flask_restx import Namespace, Resource

from app.inference.service import InputData, HorseInferenceService, OrangeInferenceService, AppleInferenceService, \
    ZebraInferenceService

api = Namespace('Inference')


class InferenceResource(Resource):
    inference_service = None

    def post(self):
        request_data = InputData(request)
        inference_result = self.inference_service.infer(request_data)
        response_data = inference_result.prepare_to_send()
        return send_file(response_data, mimetype='image/PNG')


@api.route('/horse', methods=['POST'])
class HorseInferenceResource(InferenceResource):
    inference_service = HorseInferenceService()


@api.route('/zebra', methods=['POST'])
class ZebraInferenceResource(InferenceResource):
    inference_service = ZebraInferenceService()


@api.route('/orange', methods=['POST'])
class OrangeInferenceResource(InferenceResource):
    inference_service = OrangeInferenceService()


@api.route('/apple', methods=['POST'])
class AppleInferenceResource(InferenceResource):
    inference_service = AppleInferenceService()
