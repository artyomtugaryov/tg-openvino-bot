import os
from typing import List

import cv2
import numpy as np
from openvino.inference_engine import IECore

from app.constants import MODELS_PATH
# from app.inference.input_data import GenderTransformerInputData as InputData
from app.inference.inference_result import InferenceResult
from app.inference.schema.input_info import SingleInputInfo, InputType, InputInfo


class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMetaClass, cls).__call__(*args, *kwargs)
        return cls._instances[cls]


class InferenceService(metaclass=SingletonMetaClass):
    ie_core = IECore()

    def __init__(self, device: str = 'CPU'):
        self._network = self.ie_core.read_network(model=self.network_path)
        self._execution_network = self.ie_core.load_network(network=self._network,
                                                            device_name=device)
        self._input_data = None
        self._input_info = self._define_input_info()
        self._output_blob = next(iter(self._network.outputs))

    def infer(self) -> InferenceResult:
        results = self._execution_network.infer(inputs=self.input_data)
        result = results[self._output_blob]
        return InferenceResult(result)

    def _define_input_info(self) -> InputInfo:
        raise NotImplementedError

    @property
    def network_path(self) -> str:
        raise NotImplementedError

    @property
    def input_data(self) -> dict:
        return self._input_data

    @input_data.setter
    def input_data(self, input_data: dict):
        self._input_data = input_data

    @property
    def input_info(self) -> InputInfo:
        return self._input_info


class GenderInferenceService(InferenceService):

    def _define_input_info(self) -> InputInfo:
        inputs = InputInfo()
        for input_blob_name, input_info_data in self._network.input_info.items():

            input_shape = self._network.input_info[input_blob_name].input_data.shape
            input_type = None

            if input_info_data.layout == 'NCHW':
                input_type = InputType.image
            elif input_info_data.layout == 'NC':
                input_type = InputType.labels

            inputs.add_input(
                input_type=input_type,
                single_input_info=SingleInputInfo(name=input_blob_name,
                                                  shape=input_shape)
            )
        return inputs

    @property
    def network_path(self) -> str:
        return os.path.join(MODELS_PATH, 'gender', 'gender_transformer.xml')
