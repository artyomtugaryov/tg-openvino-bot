from typing import Dict

import numpy as np
from openvino.inference_engine import IECore

from services.inference.data import Data


class IEngine:
    def __init__(self, ):
        self._ie_core = IECore()
        self._network = self._ie_core.read_network(model=self.network_path)

        self._output_blobs = set()
        for output in self._network.outputs:
            self._output_blobs.add(output)

        self._execution_network = None

    def infer(self, data: Data) -> Data:
        return Data(self._raw_inference(data.to_inference()))

    def _raw_inference(self, data: Dict[str, np.ndarray]):
        return self._execution_network.infer(inputs=data)

    @property
    def network_path(self) -> str:
        raise NotImplementedError
