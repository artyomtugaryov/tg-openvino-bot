import re
from enum import Enum
from typing import Dict

import numpy as np

from services.inference.data import InputData, InferenceResultData
from services.inference.input_utils import InputType


class FaceDetectionInputData(InputData):
    def __init__(self, image_data):
        super().__init__()
        self.add_input(input_type=InputType.image, data=image_data)


class FaceDetectionOutputType(Enum):
    boxes = 'boxes'
    scores = 'scores'


class FaceDetectionInferenceResultData(InferenceResultData):

    def _parse_output_data(self, raw_inference_results: Dict[str, np.ndarray]) -> Dict[
        FaceDetectionOutputType, np.ndarray]:
        boxes_output = next(
            raw_inference_results[name][0] for name in raw_inference_results if re.search('.bbox.', name)
        )
        scores_output = next(
            raw_inference_results[name][0] for name in raw_inference_results if re.search('.cls.', name)
        )
        return {
            FaceDetectionOutputType.boxes: boxes_output,
            FaceDetectionOutputType.scores: scores_output
        }

    @property
    def boxes(self) -> np.ndarray:
        return self._data[FaceDetectionOutputType.boxes]

    @property
    def scores(self) -> np.ndarray:
        return self._data[FaceDetectionOutputType.scores]
