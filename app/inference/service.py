import os

import cv2
import numpy as np
from openvino.inference_engine import IECore

from app.constants import MODELS_PATH
from app.inference.input_data import InputData
from app.inference.inference_result import InferenceResult


class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMetaClass, cls).__call__(*args, *kwargs)
        return cls._instances[cls]


class InferenceService(metaclass=SingletonMetaClass):
    ie_core = IECore()

    def __init__(self, network_path: str):
        network = self.ie_core.read_network(model=network_path)

        self.input_blob = next(iter(network.input_info))
        self.input_shape = network.input_info[self.input_blob].input_data.shape
        self.output_blob = next(iter(network.outputs))

        self.execution_network = self.ie_core.load_network(network=network, device_name='CPU')

    def infer(self, data: InputData) -> InferenceResult:
        input_data = data.prepare(self.input_shape)
        results = self.execution_network.infer(inputs={self.input_blob: input_data})
        result = results[self.output_blob]
        return InferenceResult(result)


class ZebraInferenceService(InferenceService):
    def __init__(self):
        super().__init__(os.path.join(MODELS_PATH, '2zebra', 'horse2zebra.xml'))


class HorseInferenceService(InferenceService):
    def __init__(self):
        super().__init__(os.path.join(MODELS_PATH, '2horse', 'zebra2horse.xml'))


class OrangeInferenceService(InferenceService):
    def __init__(self):
        super().__init__(os.path.join(MODELS_PATH, '2orange', 'apple2orange.xml'))


class AppleInferenceService(InferenceService):
    def __init__(self):
        super().__init__(os.path.join(MODELS_PATH, '2apple', 'orange2apple.xml'))
