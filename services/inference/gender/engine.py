import os
from typing import Dict, Generic

import numpy as np

from constants import MODELS_PATH
from services.inference import Data, IEngine
from services.inference.gender.data import GenderInputData
from services.inference.gender.input_schema import GenderInputInfo


class GenderEngine(IEngine):
    input_info_class = GenderInputInfo

    def infer(self, data: GenderInputData) -> Data:
        self._reshape_network(data.shape)
        raw_inference_result = self._raw_inference(data.to_infer)
        return Data(raw_inference_result)

    def _reshape_network(self, shapes: Dict[str, np.ndarray]):
        self._network.reshape(shapes)
        self._execution_network = self._ie_core.load_network(self._network, 'CPU')

    @property
    def network_path(self) -> str:
        return os.path.join(MODELS_PATH, 'gender', 'gender_transformer.xml')
