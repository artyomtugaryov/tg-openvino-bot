from typing import Dict, Generic

import numpy as np
from openvino.inference_engine import IECore

from services.inference.data import Data
from services.inference.input_schema import InputSchema, InputInfo


class IEngine:
    input_info_class = InputInfo

    def __init__(self, ):
        self._ie_core = IECore()
        self._network = self._ie_core.read_network(model=self.network_path)

        self._output_blobs = set()
        for output in self._network.outputs:
            self._output_blobs.add(output)

        self._execution_network = None

    def infer(self, data: Data) -> Data:
        raise NotImplementedError

    def _raw_inference(self, data: Dict[str, np.ndarray]):
        return self._execution_network.infer(inputs=data)

    @property
    def network_path(self) -> str:
        raise NotImplementedError

    @property
    def input_schema(self) -> InputSchema:
        inputs = []
        for input_blob_name, input_info_data in self._network.input_info.items():
            input_type = self.input_info_class.define_input_type(input_info_data)
            inputs.append(self.input_info_class(name=input_blob_name, input_type=input_type))

        return InputSchema(inputs)
