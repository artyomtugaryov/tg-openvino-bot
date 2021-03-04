import os
from typing import Dict

import numpy as np

from constants import MODELS_PATH
from services.inference.data import Data
from services.inference.engine import IEngine
from services.inference.gender.data import CompoundInputData


class GenderEngine(IEngine):

    def infer(self, data: CompoundInputData) -> Data:
        shapes = self.combine(self.input_schema(), data.shape)
        self._reshape_network(shapes)
        data = self.combine(self.input_schema(), data.to_infer)
        raw_inference_result = self._raw_inference(data)
        return Data(raw_inference_result)

    def _reshape_network(self, shapes: Dict[str, np.ndarray]):
        self._network.reshape(shapes)
        self._execution_network = self._ie_core.load_network(self._network, 'CPU')

    @property
    def network_path(self) -> str:
        return os.path.join(MODELS_PATH, 'gender', 'gender_transformer.xml')

    def input_schema(self) -> Dict[str, str]:
        inputs = {}
        for input_blob_name, input_info_data in self._network.input_info.items():

            if input_info_data.layout == 'NCHW':
                inputs['image'] = input_blob_name
            elif input_info_data.layout == 'NC':
                inputs['flags'] = input_blob_name
            else:
                raise AssertionError('Unsupported type of input')

        return inputs

    @staticmethod
    def combine(input_info, data):
        return {
            input_info[key]: value
            for key, value in data.items()
        }
