from typing import Dict

import numpy as np
from openvino.inference_engine import IECore

from services.inference.data import Data, InputData
from services.inference.input_utils import InputType, define_input_type


class IEngine:
    def __init__(self):
        self._ie_core = IECore()
        self._network = self._ie_core.read_network(model=self.network_path)

        self._inputs = self._define_inputs()

        self._output_blobs = set()
        for output in self._network.outputs:
            self._output_blobs.add(output)

        self._execution_network = None

    def _define_inputs(self) -> Dict[InputType, str]:
        return {
            define_input_type(input_info): input_name
            for input_name, input_info in self._network.input_info.items()
        }

    def _prepare_data(self, data: InputData) -> Dict[str, np.ndarray]:
        return {
            input_name: data.raw_data[input_type]
            for input_type, input_name in self._inputs.items()
        }

    def infer(self, data: Data) -> Data:
        raise NotImplementedError

    @property
    def network_path(self) -> str:
        raise NotImplementedError

    def _raw_inference(self, data: Dict[str, np.ndarray]):
        return self._execution_network.infer(inputs=data)
