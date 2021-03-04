from enum import Enum
from enum import Enum
from typing import Generic
from typing import List, TypeVar


class InputLayout(Enum):
    nchw = 'NCHW'


class InputType(Enum):
    image = 'image'
    undefined = 'undefined'


class InputInfo:
    def __init__(self, name: str, input_type):
        self.name = name
        self.input_type = input_type

    @staticmethod
    def define_input_type(input_info_data) -> InputType:
        if input_info_data.layout == InputLayout.nchw:
            return InputType.image
        return InputType.undefined


class InputSchema:
    def __init__(self, input_infos: List):
        self._input_infos = {
            input_info.input_type: input_info.name
            for input_info in input_infos
        }

    def layer_name_by_type(self, input_type: InputType) -> str:
        return self._input_infos[input_type]
