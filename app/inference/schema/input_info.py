from enum import Enum
from typing import Tuple
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
        self.shape = shape
        self.data = data


class InputInfo:
    def __init__(self):
        self._inputs = {}

    def add_input(self, input_type: InputType = InputType.undefined,
                  single_input_info: SingleInputInfo = None):
        self._inputs[input_type] = single_input_info

    def get_input(self, input_type: InputType) -> SingleInputInfo:
        return self._inputs[input_type]