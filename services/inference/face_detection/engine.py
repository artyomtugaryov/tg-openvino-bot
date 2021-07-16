import os
from typing import Dict

import numpy as np

from constants import MODELS_PATH
from services.inference import Data, IEngine
from services.inference.face_detection import FaceDetectionInputData
from services.inference.face_detection.data import FaceDetectionInferenceResultData


class FaceDetectionEngine(IEngine):

    def infer(self, data: FaceDetectionInputData) -> FaceDetectionInferenceResultData:
        data_to_infer = self._prepare_data(data)
        raw_inference_result = self._raw_inference(data_to_infer)
        return FaceDetectionInferenceResultData(raw_inference_result)

    @property
    def network_path(self) -> str:
        return os.path.join(MODELS_PATH, 'retinaface-resnet50-pytorch.xml')
