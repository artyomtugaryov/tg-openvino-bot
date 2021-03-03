from enum import Enum
from typing import Tuple, List, Dict, Optional
import numpy as np


class InputType(Enum):
    image = 'image'
    labels = 'labels'
    undefined = 'undefined'


class SingleInputInfo:
    def __init__(self, name: str = None,
                 shape: Tuple[int, ...] = None,
                 data: np.ndarray = None):
        self.name = name
        self._shape = shape
        self._data = data

    @property
    def data(self) -> Optional[np.ndarray]:
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        self._data = data
        self._shape = self._data.shape

    @property
    def shape(self) -> Optional[np.ndarray]:
        return self._shape


class InputInfo:
    def __init__(self):
        self._inputs = {}

    def add_input(self, input_type: InputType = InputType.undefined,
                  single_input_info: SingleInputInfo = None):
        self._inputs[input_type] = single_input_info

    def get_input(self, input_type: InputType) -> SingleInputInfo:
        return self._inputs[input_type]

    @property
    def for_inference(self) -> Dict[str, np.ndarray]:
        return {
            single_input_info.name: single_input_info.data
            for single_input_info in self.inputs
        }

    @property
    def shapes(self) -> dict:
        return {
            single_input_info.name: single_input_info.shape
            for single_input_info in self.inputs
        }

    @property
    def inputs(self) -> List[SingleInputInfo]:
        return list(self._inputs.values())
