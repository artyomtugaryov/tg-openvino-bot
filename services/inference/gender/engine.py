import os
from typing import Dict

import numpy as np

from constants import MODELS_PATH
from services.inference import Data, IEngine
from services.inference.data import InputData
from services.inference.gender.data import GenderInputData


class GenderEngine(IEngine):

    def infer(self, data: GenderInputData) -> Data:
        new_shapes = self._prepare_shape(data)
        self._reshape_network(new_shapes)
        data_to_infer = self._prepare_data(data)
        raw_inference_result = self._raw_inference(data_to_infer)
        return Data(raw_inference_result)

    def _reshape_network(self, shapes: Dict[str, np.ndarray]):
        self._network.reshape(shapes)
        self._execution_network = self._ie_core.load_network(self._network, 'CPU')

    def _prepare_shape(self, data: InputData) -> Dict[str, np.ndarray]:
        return {
            input_name: data.shape[input_type]
            for input_type, input_name in self._inputs.items()
        }

    @property
    def network_path(self) -> str:
        return os.path.join(MODELS_PATH, 'gender', 'gender_transformer.xml')
